"""
GameConfig: serializable full game configuration for the optimizer.

Wraps all existing settings abstractions (GameSettings, GameMechanics) and
captures the complete board state: cell layout, chance/chest decks.

Use to_dict() / from_dict() to interface with the optimization loop.
Use from_board() to snapshot a live Board + its GameSettings.
"""
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Tuple
import yaml

from settings import GameSettings, GameMechanics
from player_settings import (StandardPlayerSettings, HeroPlayerSettings,
                              RuleBasedPlayerSettings, RandomPlayerSettings,
                              ParametricPlayerSettings)
from agents import RandomPlayer, ParametricPlayer, LLMPlayer
from ladder_agents import MonopolyTargeterPlayer, GeneralistPlayer

# Maps settings class name -> class, for YAML player deserialization
_PLAYER_SETTINGS_CLASSES = {
    'StandardPlayerSettings': StandardPlayerSettings,
    'HeroPlayerSettings': HeroPlayerSettings,
    'RuleBasedPlayerSettings': RuleBasedPlayerSettings,
    'RandomPlayerSettings': RandomPlayerSettings,
    'ParametricPlayerSettings': ParametricPlayerSettings,
}

# Maps optional player_class name -> constructor, for non-default Player subclasses
_PLAYER_CLASSES = {
    'RandomPlayer':     RandomPlayer,
    'ParametricPlayer': ParametricPlayer,
    'LLMPlayer':        LLMPlayer,
    'MonopolyTargeterPlayer': MonopolyTargeterPlayer,
    'GeneralistPlayer': GeneralistPlayer,
}
from monopoly.core.cell import (
    Cell, GoToJail, LuxuryTax, IncomeTax, FreeParking,
    Chance, CommunityChest, Property,
)
from monopoly.core.deck import Deck

# Maps class name -> constructor for non-Property cells (all take only a name arg)
_CELL_CLASSES: Dict[str, type] = {
    'Cell': Cell,
    'GoToJail': GoToJail,
    'LuxuryTax': LuxuryTax,
    'IncomeTax': IncomeTax,
    'FreeParking': FreeParking,
    'Chance': Chance,
    'CommunityChest': CommunityChest,
}


@dataclass
class OptimizationSpec:
    """Declares which fields in GameConfig.to_dict() the optimizer can vary and their bounds.

    Keys match the flat dict keys produced by to_dict() (e.g. 'salary', 'prop_1_cost_base').
    Values are (lower_bound, upper_bound) as floats.
    Fields absent from bounds are treated as fixed.
    """
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def is_optimizable(self, key: str) -> bool:
        return key in self.bounds

    def parameter_space(self) -> Dict[str, Tuple[float, float]]:
        """Return the full bounds dict for passing to the optimizer."""
        return dict(self.bounds)


def _fresh_cell(cell: Cell) -> Cell:
    """Return a new Cell instance with config params only (no game state)."""
    if isinstance(cell, Property):
        return Property(cell.name, cell.cost_base, cell.rent_base,
                        cell.cost_house, cell.rent_house, cell.group)
    return deepcopy(cell)


@dataclass
class GameConfig:
    # All game/player/mechanics settings via the existing GameSettings dataclass
    settings: GameSettings = field(default_factory=GameSettings)
    # Full board layout as Cell/Property instances (config params only, no game state)
    cells: List[Cell] = field(default_factory=list)
    # Chance and Community Chest decks (card lists)
    chance: Deck = field(default_factory=lambda: Deck([]))
    chest: Deck = field(default_factory=lambda: Deck([]))
    # Which fields are optimizable and their (lo, hi) bounds
    optimization_spec: OptimizationSpec = field(default_factory=OptimizationSpec)
    # Player list: each entry is {'name': str, 'settings': class_name_str, 'starting_money': int}
    # Empty list means use GameSettings.players_list class defaults
    players: List[dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Flatten to a dict suitable for the optimizer's parameter space."""
        d = asdict(self.settings.mechanics)

        # Player starting money (other GameSettings fields are not varied by the optimizer)
        starting_money = self.settings.starting_money
        if isinstance(starting_money, dict):
            for name, amount in starting_money.items():
                d[f'starting_money_{name}'] = amount
        else:
            d['starting_money'] = starting_money

        # Board cells
        for i, cell in enumerate(self.cells):
            d[f'cell_{i}_type'] = type(cell).__name__
            d[f'cell_{i}_name'] = cell.name
            if isinstance(cell, Property):
                d[f'cell_{i}_cost_base'] = cell.cost_base
                d[f'cell_{i}_rent_base'] = cell.rent_base
                d[f'cell_{i}_cost_house'] = cell.cost_house
                d[f'cell_{i}_group'] = cell.group
                for j, r in enumerate(cell.rent_house):
                    d[f'cell_{i}_rent_house_{j}'] = r

        # Card decks
        for i, card in enumerate(self.chance.cards):
            d[f'chance_card_{i}'] = card
        for i, card in enumerate(self.chest.cards):
            d[f'chest_card_{i}'] = card

        # Optimization spec bounds
        for key, (lo, hi) in self.optimization_spec.bounds.items():
            d[f'opt_lo_{key}'] = lo
            d[f'opt_hi_{key}'] = hi

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GameConfig':
        """Reconstruct a GameConfig from a flat dict (inverse of to_dict)."""
        mechanics_keys = set(GameMechanics.__dataclass_fields__)
        mechanics = GameMechanics(**{k: d[k] for k in mechanics_keys if k in d})
        settings = GameSettings(mechanics=mechanics)

        n_cells = sum(1 for k in d if k.startswith('cell_') and k.endswith('_type'))
        cells = []
        for i in range(n_cells):
            cell_type = d[f'cell_{i}_type']
            name = d[f'cell_{i}_name']
            if cell_type == 'Property':
                rent_house = tuple(d[f'cell_{i}_rent_house_{j}'] for j in range(5))
                cells.append(Property(name, d[f'cell_{i}_cost_base'],
                                      d[f'cell_{i}_rent_base'], d[f'cell_{i}_cost_house'],
                                      rent_house, d[f'cell_{i}_group']))
            else:
                cells.append(_CELL_CLASSES[cell_type](name))

        n_chance = sum(1 for k in d if k.startswith('chance_card_'))
        chance = Deck([d[f'chance_card_{i}'] for i in range(n_chance)])

        n_chest = sum(1 for k in d if k.startswith('chest_card_'))
        chest = Deck([d[f'chest_card_{i}'] for i in range(n_chest)])

        opt_keys = {key[len('opt_lo_'):] for key in d if key.startswith('opt_lo_')}
        bounds = {key: (d[f'opt_lo_{key}'], d[f'opt_hi_{key}']) for key in opt_keys}
        optimization_spec = OptimizationSpec(bounds=bounds)

        return cls(settings=settings, cells=cells, chance=chance, chest=chest,
                   optimization_spec=optimization_spec)

    @classmethod
    def from_board(cls, board) -> 'GameConfig':
        """Snapshot a Board instance (and its settings) into a GameConfig."""
        return cls(
            settings=board.settings,
            cells=[_fresh_cell(cell) for cell in board.cells],
            chance=Deck(list(board.chance.cards)),
            chest=Deck(list(board.chest.cards)),
        )

    def to_yaml(self, path: str) -> None:
        """Write this config to a YAML file."""
        mechanics = asdict(self.settings.mechanics)

        cells = []
        for cell in self.cells:
            entry = {'type': type(cell).__name__, 'name': cell.name}
            if isinstance(cell, Property):
                entry['cost_base'] = cell.cost_base
                entry['rent_base'] = cell.rent_base
                entry['cost_house'] = cell.cost_house
                entry['rent_house'] = list(cell.rent_house)
                entry['group'] = cell.group
            cells.append(entry)

        bounds = {k: list(v) for k, v in self.optimization_spec.bounds.items()}

        doc = {
            'mechanics': mechanics,
            'players': self.players,
            'cells': cells,
            'chance_cards': list(self.chance.cards),
            'chest_cards': list(self.chest.cards),
            'optimization_spec': bounds,
        }
        with open(path, 'w') as f:
            yaml.dump(doc, f, default_flow_style=False, sort_keys=False)

    def to_config_dir(self, dir_path: str) -> None:
        """Write this config to a structured directory of YAML files with a master.yaml."""
        base = Path(dir_path)
        (base / 'players').mkdir(parents=True, exist_ok=True)
        (base / 'board').mkdir(exist_ok=True)
        (base / 'properties').mkdir(exist_ok=True)
        (base / 'decks').mkdir(exist_ok=True)

        with open(base / 'mechanics.yaml', 'w') as f:
            yaml.dump(asdict(self.settings.mechanics), f, default_flow_style=False, sort_keys=False)

        with open(base / 'players/players.yaml', 'w') as f:
            yaml.dump(self.players, f, default_flow_style=False, sort_keys=False)

        layout = [{'type': type(cell).__name__, 'name': cell.name} for cell in self.cells]
        with open(base / 'board/layout.yaml', 'w') as f:
            yaml.dump(layout, f, default_flow_style=False, sort_keys=False)

        props = [
            {'name': cell.name, 'cost_base': cell.cost_base, 'rent_base': cell.rent_base,
             'cost_house': cell.cost_house, 'rent_house': list(cell.rent_house), 'group': cell.group}
            for cell in self.cells if isinstance(cell, Property)
        ]
        with open(base / 'properties/properties.yaml', 'w') as f:
            yaml.dump(props, f, default_flow_style=False, sort_keys=False)

        with open(base / 'decks/chance.yaml', 'w') as f:
            yaml.dump(list(self.chance.cards), f, default_flow_style=False, sort_keys=False)
        with open(base / 'decks/chest.yaml', 'w') as f:
            yaml.dump(list(self.chest.cards), f, default_flow_style=False, sort_keys=False)

        bounds = {k: list(v) for k, v in self.optimization_spec.bounds.items()}
        with open(base / 'optimization.yaml', 'w') as f:
            yaml.dump(bounds, f, default_flow_style=False, sort_keys=False)

        master = {
            'mechanics': 'mechanics.yaml',
            'players': 'players/players.yaml',
            'layout': 'board/layout.yaml',
            'properties': 'properties/properties.yaml',
            'chance_deck': 'decks/chance.yaml',
            'chest_deck': 'decks/chest.yaml',
            'optimization': 'optimization.yaml',
        }
        with open(base / 'master.yaml', 'w') as f:
            yaml.dump(master, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'GameConfig':
        """Load a GameConfig from a flat YAML file or a master.yaml in a config directory."""
        path = Path(path)
        # If given a directory, look for master.yaml inside it
        if path.is_dir():
            path = path / 'master.yaml'
        base_dir = path.parent

        with open(path, 'r') as f:
            doc = yaml.safe_load(f)

        def _load(value):
            """Resolve a string file reference relative to base_dir, or return value as-is."""
            if isinstance(value, str) and value.endswith('.yaml'):
                with open(base_dir / value, 'r') as f:
                    return yaml.safe_load(f)
            return value

        # Mechanics
        mechanics = GameMechanics(**_load(doc['mechanics']))
        settings = GameSettings(mechanics=mechanics)

        # Board: merge layout (type + name) with property params (keyed by name)
        if 'layout' in doc:
            layout = _load(doc['layout'])
            props_by_name = {p['name']: p for p in _load(doc['properties'])}
            cells = []
            for entry in layout:
                cell_type, name = entry['type'], entry['name']
                if cell_type == 'Property':
                    p = props_by_name[name]
                    cells.append(Property(name, p['cost_base'], p['rent_base'],
                                          p['cost_house'], tuple(p['rent_house']), p['group']))
                else:
                    cells.append(_CELL_CLASSES[cell_type](name))
        else:
            # Flat format: cells inline
            cells = []
            for entry in doc['cells']:
                cell_type, name = entry['type'], entry['name']
                if cell_type == 'Property':
                    cells.append(Property(name, entry['cost_base'], entry['rent_base'],
                                          entry['cost_house'], tuple(entry['rent_house']), entry['group']))
                else:
                    cells.append(_CELL_CLASSES[cell_type](name))

        chance = Deck(_load(doc.get('chance_deck', doc.get('chance_cards', []))) or [])
        chest = Deck(_load(doc.get('chest_deck', doc.get('chest_cards', []))) or [])

        bounds = {k: tuple(v) for k, v in (_load(doc.get('optimization', doc.get('optimization_spec', {}))) or {}).items()}
        optimization_spec = OptimizationSpec(bounds=bounds)

        players = _load(doc.get('players', []))

        return cls(settings=settings, cells=cells, chance=chance, chest=chest,
                   optimization_spec=optimization_spec, players=players or [])
