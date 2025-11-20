# Make agents a package
from .plant_agent import PlantAgent
from .filter_agent import FilterAgent
from .docking_agent import SimpleDockingAgent, setup_all_targets_rapid
from .optimization_agent import MoleculeOptimizationAgent

__all__ = [
    'PlantAgent',
    'FilterAgent', 
    'SimpleDockingAgent',
    'MoleculeOptimizationAgent',
    'setup_all_targets_rapid'
]
