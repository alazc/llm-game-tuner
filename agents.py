"""Agent population for the Monopoly optimizer.

Agents in increasing order of skill:
  RandomPlayer    -- buys each unowned property with fixed probability
  RuleBasedPlayer -- always buy, always build (via RuleBasedPlayerSettings in settings.py)

Settings classes live in settings.py alongside StandardPlayerSettings.
Player subclasses (requiring custom logic beyond settings) live here.
"""
import os
import random as _random

from monopoly.core.constants import RAILROADS, UTILITIES
from monopoly.core.player import Player
from player_settings import ParametricPlayerSettings, RandomPlayerSettings


class ParametricPlayer(Player):
    """Rule-based player with the full parametric knob-set for the optimiser.

    Every decision point the optimiser might want to vary is gated by a field on
    ``ParametricPlayerSettings``. Behaviours not yet expressible in the base
    Player (aggressive-vs-one-at-a-time building, skip-utilities/skip-railroads,
    early jail exit) are implemented here; everything else (trading thresholds,
    unspendable_cash, ignore_property_groups) is inherited from
    StandardPlayerSettings semantics via base-class logic.
    """

    def __init__(self, name: str, settings=None):
        super().__init__(name, settings or ParametricPlayerSettings())

    # ------------------------------------------------------------------ #
    # Buy decision                                                          #
    # ------------------------------------------------------------------ #
    def _should_buy(self, property_to_buy) -> bool:
        # Base affordability + ignored-groups check from parent
        if property_to_buy.cost_base > self.money:
            return False
        if self.money - property_to_buy.cost_base < self.settings.unspendable_cash:
            return False
        if property_to_buy.group in self.settings.ignore_property_groups:
            return False
        # Parametric knobs
        if property_to_buy.group == UTILITIES and not getattr(
                self.settings, 'buy_utilities', True):
            return False
        if property_to_buy.group == RAILROADS and not getattr(
                self.settings, 'buy_railroads', True):
            return False
        return True

    # ------------------------------------------------------------------ #
    # Build decision                                                        #
    # ------------------------------------------------------------------ #
    @property
    def _improve_cash_floor(self) -> int:
        return getattr(self.settings, 'build_cash_floor',
                       self.settings.unspendable_cash)

    def improve_properties(self, board, log):
        """If aggressive_build is False, build at most 1 house per turn.

        This is a targeted override of ``Player.improve_properties`` (core/player.py).
        When aggressive, we just delegate to super(). Otherwise we inline the
        one-iteration version, kept in sync with core/player.py:528-602.
        """
        if getattr(self.settings, 'aggressive_build', True):
            return super().improve_properties(board, log)

        # Non-aggressive: exactly one build attempt per turn.
        from monopoly.core.constants import RAILROADS as _RR, UTILITIES as _UT
        can_be_improved = []
        for cell in self.owned:
            if (cell.has_hotel == 0
                    and not cell.is_mortgaged
                    and cell.monopoly_multiplier == 2
                    and cell.group not in (_RR, _UT)):
                eligible = True
                for other_cell in board.groups[cell.group]:
                    if ((other_cell.has_houses < cell.has_houses
                         and not other_cell.has_hotel)
                            or other_cell.is_mortgaged):
                        eligible = False
                        break
                if eligible:
                    if cell.has_houses != 4 and board.available_houses > 0 or \
                            cell.has_houses == 4 and board.available_hotels > 0:
                        can_be_improved.append(cell)
        if not can_be_improved:
            return
        can_be_improved.sort(key=lambda x: x.cost_house)
        cell_to_improve = can_be_improved[0]
        if self.money - cell_to_improve.cost_house < self._improve_cash_floor:
            return
        ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
        if cell_to_improve.has_houses != 4:
            cell_to_improve.has_houses += 1
            board.available_houses -= 1
            self.money -= cell_to_improve.cost_house
            log.add(f"{self} built {ordinal[cell_to_improve.has_houses]} "
                    f"house on {cell_to_improve} for ${cell_to_improve.cost_house}")
        else:
            cell_to_improve.has_houses = 0
            cell_to_improve.has_hotel = 1
            board.available_houses += 4
            board.available_hotels -= 1
            self.money -= cell_to_improve.cost_house
            log.add(f"{self} built a hotel on {cell_to_improve}")

    # ------------------------------------------------------------------ #
    # Jail decision                                                         #
    # ------------------------------------------------------------------ #
    def handle_jail(self, dice_roll_is_double, board, log):
        """Extend base jail handling with an early-pay option.

        Parent (`core/player.py:handle_jail`) only pays the fine after the
        mandatory 2-turn wait. If ``jail_pay_threshold`` is set and the player
        has comfortably more cash than that threshold, pay on day 0 or 1 too
        — it's strategically useful when opponents have few developed cells.
        """
        from settings import GameMechanics
        threshold = getattr(self.settings, 'jail_pay_threshold', None)
        if (threshold is not None
                and self.in_jail
                and not self.get_out_of_jail_chance
                and not self.get_out_of_jail_comm_chest
                and not dice_roll_is_double
                and self.days_in_jail < 2
                and self.money > threshold + GameMechanics.exit_jail_fine):
            log.add(f"{self} chooses to pay ${GameMechanics.exit_jail_fine} "
                    f"and exit jail early (cash > threshold={threshold})")
            self.pay_money(GameMechanics.exit_jail_fine, "bank", board, log)
            self.in_jail = False
            self.days_in_jail = 0
            return False
        return super().handle_jail(dice_roll_is_double, board, log)


class LLMPlayer(Player):
    """Monopoly player whose buy decisions are queried from an LLM.

    Default backend: a local Qwen2.5-1.5B-Instruct model via huggingface
    transformers. The model is loaded lazily on the first decision so
    constructing an LLMPlayer is cheap, and is shared across instances
    via a module-level cache so multi-game runs don't re-load weights.

    For build / trade / jail decisions we fall back to base-Player logic:
    LLM latency on every micro-decision would make games take minutes
    instead of milliseconds, and the buy decision is the strategically
    decisive one anyway.

    Backend selection (decided at construction):
      - backend='local'     (default): use transformers + Qwen locally.
      - backend='openai'    : POST to an OpenAI-compatible endpoint.
      - backend='heuristic' : skip the LLM entirely; same as RuleBasedPlayer
        with extra logging hooks. Useful for unit tests.

    Environment variables (only read in the matching backend):
      LLM_MODEL            (default: 'Qwen/Qwen2.5-1.5B-Instruct')
      LLM_CACHE_DIR        Folder for HF model weights. Default: 'models/hf_cache'
                           relative to CWD, so weights land inside the project
                           rather than the user's global ~/.cache/huggingface.
      LLM_OPENAI_BASE_URL  (e.g. 'http://localhost:11434/v1' for ollama)
      LLM_OPENAI_API_KEY
      LLM_OPENAI_MODEL     (e.g. 'qwen2.5:1.5b' for ollama)
    """

    _MODEL_CACHE: dict = {}

    def __init__(self, name: str, settings=None, backend: str = None,
                 model_name: str = None, max_new_tokens: int = 64):
        from player_settings import StandardPlayerSettings
        super().__init__(name, settings or StandardPlayerSettings())
        self._backend = backend or os.environ.get('LLM_BACKEND', 'local')
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        # Per-instance counter for diagnostic logging.
        self._n_buy_decisions = 0
        self._n_buy_yes = 0
        # Refs cached at the start of each turn by make_a_move so the buy
        # prompt can reference board state and opponents.
        self._board_ref = None
        self._players_ref = None

    def make_a_move(self, board, players, dice, log):
        # Cache refs so _should_buy / _build_buy_prompt can include
        # board state in the LLM query.
        self._board_ref = board
        self._players_ref = players
        return super().make_a_move(board, players, dice, log)

    # ------------------------------------------------------------------ #
    # Backend                                                              #
    # ------------------------------------------------------------------ #

    def _get_local_model(self):
        import os
        from pathlib import Path
        model_name = self._model_name or os.environ.get(
            'LLM_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct')
        if model_name in self._MODEL_CACHE:
            return self._MODEL_CACHE[model_name]
        # If model_name looks like a path on disk, load from there directly
        # (no HF hub interaction). Otherwise treat it as a hub repo id and
        # use a project-local HF cache so weights don't pollute the global
        # ~/.cache/huggingface and are easy to wipe / .gitignore.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        is_local = Path(model_name).is_dir()
        if is_local:
            from_pretrained_kw = {}
        else:
            cache_dir = os.environ.get('LLM_CACHE_DIR', 'models/hf_cache')
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            from_pretrained_kw = {'cache_dir': cache_dir}
        tok = AutoTokenizer.from_pretrained(model_name, **from_pretrained_kw)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Precision + attention impl can be overridden via env vars so the
        # cluster can opt into bf16 + flash-attn-2 without changing default
        # behavior on dev hosts (where Phase A's original runs were captured).
        kw = {}
        if device == 'cuda':
            dtype_name = os.environ.get('LLM_DTYPE', 'float16')
            dtype = {'float16': torch.float16,
                      'bfloat16': torch.bfloat16,
                      'float32':  torch.float32}.get(dtype_name, torch.float16)
            kw['torch_dtype'] = dtype
            attn_impl = os.environ.get('LLM_ATTN_IMPL')
            if attn_impl:
                kw['attn_implementation'] = attn_impl
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **from_pretrained_kw, **kw).to(device)
        model.eval()
        self._MODEL_CACHE[model_name] = (tok, model, device)
        return self._MODEL_CACHE[model_name]

    # ------------------------------------------------------------------ #
    # Prompting                                                            #
    # ------------------------------------------------------------------ #

    _SYSTEM_PROMPT = (
        "You are a strategic Monopoly player who decides whether to buy "
        "properties. Your goal is to win by bankrupting opponents through "
        "rent. You must NOT mindlessly buy every property: passing is "
        "correct when (a) cash is dangerously low, (b) the property is "
        "in a group an opponent already dominates and you have no path "
        "to monopoly, or (c) the rent return is poor relative to the "
        "cost. Equally, buying is correct when the property advances a "
        "group you already own most of, or when it denies the group to "
        "an opponent who would otherwise complete it.\n\n"
        "Reply in EXACTLY this format on a single line:\n"
        "  REASON: <one short sentence>\n"
        "  ANSWER: BUY\n"
        "or\n"
        "  REASON: <one short sentence>\n"
        "  ANSWER: PASS\n"
        "Do not output anything else."
    )

    _FEW_SHOT = [
        {"role": "user", "content":
            "Cash $1200. Decision: buy 'B1 Oriental Avenue' (group Lightblue) "
            "for $100, base rent $6. You own 0 properties total, 0 in this "
            "group. Opponent owns 0 in this group, 2 properties total."},
        {"role": "assistant", "content":
            "REASON: Cheap, plenty of cash, and no opponent ownership in "
            "this group leaves room for me to monopolise it.\nANSWER: BUY"},
        {"role": "user", "content":
            "Cash $90. Decision: buy 'D1 St. James Place' (group Orange) for "
            "$180, base rent $14. You own 3 properties total, 0 in this "
            "group. Opponent owns 1 in this group, 5 properties total."},
        {"role": "assistant", "content":
            "REASON: Cash is too low to afford this and the opponent already "
            "holds Orange, so this group is unlikely to become my monopoly.\n"
            "ANSWER: PASS"},
        {"role": "user", "content":
            "Cash $700. Decision: buy 'C2 States Avenue' (group Pink) for "
            "$140, base rent $10. You own 6 properties total, 2 in this "
            "group. Opponent owns 0 in this group, 4 properties total."},
        {"role": "assistant", "content":
            "REASON: Buying this completes my Pink monopoly, which is the "
            "single highest-value move available.\nANSWER: BUY"},
        {"role": "user", "content":
            "Cash $500. Decision: buy 'G1 Pacific Avenue' (group Green) for "
            "$300, base rent $26. You own 8 properties total, 0 in this "
            "group. Opponent owns 2 in this group, 5 properties total."},
        {"role": "assistant", "content":
            "REASON: Opponent already owns two-thirds of Green so I cannot "
            "monopolise it; spending $300 here hurts my liquidity.\nANSWER: PASS"},
    ]

    def _query_local(self, prompt: str) -> str:
        import torch
        tok, model, device = self._get_local_model()
        msgs = [{'role': 'system', 'content': self._SYSTEM_PROMPT}]
        msgs.extend(self._FEW_SHOT)
        msgs.append({'role': 'user', 'content': prompt})
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors='pt').to(device)
        # Stop as soon as the model emits the chat-template's user-turn
        # marker (so it can't hallucinate a fake new user message after
        # its answer). Falling back to eos for older templates.
        eos_ids = [tok.eos_token_id]
        for marker in ('<|im_end|>', '<|endoftext|>'):
            tid = tok.convert_tokens_to_ids(marker)
            if tid is not None and tid != tok.unk_token_id:
                eos_ids.append(tid)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=eos_ids,
            )
        gen = tok.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return gen.strip()

    def _query_anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic()
        model = os.environ.get('LLM_ANTHROPIC_MODEL', 'claude-sonnet-4-20250514')
        msgs = list(self._FEW_SHOT) + [{'role': 'user', 'content': prompt}]
        resp = client.messages.create(
            model=model,
            max_tokens=self._max_new_tokens,
            temperature=0.0,
            system=self._SYSTEM_PROMPT,
            messages=msgs,
        )
        return resp.content[0].text.strip()

    def _query_openai(self, prompt: str) -> str:
        import os, json, urllib.request
        base = os.environ.get('LLM_OPENAI_BASE_URL', 'http://localhost:11434/v1')
        key = os.environ.get('LLM_OPENAI_API_KEY', 'no-key')
        model = os.environ.get('LLM_OPENAI_MODEL', 'qwen2.5:1.5b')
        msgs = [{'role': 'system', 'content': self._SYSTEM_PROMPT}]
        msgs.extend(self._FEW_SHOT)
        msgs.append({'role': 'user', 'content': prompt})
        body = {
            'model': model,
            'messages': msgs,
            'max_tokens': self._max_new_tokens,
            'temperature': 0.0,
        }
        req = urllib.request.Request(
            f'{base}/chat/completions',
            data=json.dumps(body).encode('utf-8'),
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {key}'},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
        return payload['choices'][0]['message']['content'].strip()

    # ------------------------------------------------------------------ #
    # Decision                                                              #
    # ------------------------------------------------------------------ #

    def _build_buy_prompt(self, prop) -> str:
        owned_count = len(self.owned)
        same_group_self = sum(1 for c in self.owned if c.group == prop.group)
        # Opponent context (if board/players are cached by make_a_move)
        opp_in_group = 0
        opp_total = 0
        group_size = 0
        if self._board_ref is not None and self._players_ref is not None:
            from monopoly.core.cell import Property
            group_cells = [c for c in self._board_ref.cells
                           if isinstance(c, Property) and c.group == prop.group]
            group_size = len(group_cells)
            for other in self._players_ref:
                if other is self:
                    continue
                opp_in_group += sum(1 for c in other.owned if c.group == prop.group)
                opp_total    += len(other.owned)
        return (
            f"Cash ${self.money}. Decision: buy '{prop.name}' "
            f"(group {prop.group}) for ${prop.cost_base}, base rent "
            f"${prop.rent_base}. You own {owned_count} properties total, "
            f"{same_group_self} in this group of {group_size}. "
            f"Opponent owns {opp_in_group} in this group, {opp_total} "
            f"properties total."
        )

    @staticmethod
    def _parse_decision(response: str) -> bool:
        """Extract the LLM's BUY/PASS decision from the structured response.

        We look for the FIRST occurrence of 'ANSWER: BUY' or 'ANSWER: PASS'
        (case-insensitive) because small models often hallucinate a fake
        new user turn after their answer, and that hallucinated text can
        contain the word 'buy' or 'pass' unrelated to the actual decision.
        Falls back to a last-token heuristic if neither structured form is
        found, then defaults to True (BUY) so a malformed response leaves
        the player playing roughly like a RuleBasedPlayer.
        """
        upper = response.upper()
        buy_idx  = upper.find('ANSWER: BUY')
        pass_idx = upper.find('ANSWER: PASS')
        if buy_idx >= 0 and (pass_idx < 0 or buy_idx < pass_idx):
            return True
        if pass_idx >= 0 and (buy_idx < 0 or pass_idx < buy_idx):
            return False
        # Fallback: no structured ANSWER tag — use last-token heuristic.
        last_buy  = upper.rfind('BUY')
        last_pass = upper.rfind('PASS')
        if last_buy < 0 and last_pass < 0:
            return True
        return last_buy > last_pass

    def _should_buy(self, property_to_buy) -> bool:
        # Cheap pre-filter that avoids the LLM for trivially-impossible buys.
        if property_to_buy.cost_base > self.money:
            return False
        if self.money - property_to_buy.cost_base < self.settings.unspendable_cash:
            return False
        if property_to_buy.group in self.settings.ignore_property_groups:
            return False
        # Heuristic backend: shortcut to "always buy when affordable".
        if self._backend == 'heuristic':
            self._n_buy_decisions += 1
            self._n_buy_yes += 1
            return True

        prompt = self._build_buy_prompt(property_to_buy)
        try:
            if self._backend == 'openai':
                response = self._query_openai(prompt)
            elif self._backend == 'anthropic':
                response = self._query_anthropic(prompt)
            else:
                response = self._query_local(prompt)
        except Exception:
            response = 'BUY'
        decision = self._parse_decision(response)
        self._n_buy_decisions += 1
        if decision:
            self._n_buy_yes += 1
        return decision


class RandomPlayer(Player):
    """Buys each unowned property it can afford with probability *buy_probability*.

    All other behaviour (building, trading, bankruptcy) inherits from Player.
    """

    def __init__(self, name: str, settings=None, buy_probability: float = 0.5,
                 seed: int = None):
        super().__init__(name, settings or RandomPlayerSettings())
        self._rng = _random.Random(seed)
        self._buy_probability = buy_probability

    def _should_buy(self, property_to_buy) -> bool:
        if property_to_buy.cost_base > self.money:
            return False
        return self._rng.random() < self._buy_probability
