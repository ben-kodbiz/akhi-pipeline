#!/usr/bin/env python3
"""
CrewAI Integration Framework

This module provides the core framework for enhanced CrewAI tool integration
and agent coordination. It includes:

- Tool Registry: Centralized tool discovery, registration, and management
- Agent Coordinator: Enhanced agent coordination with advanced scheduling
- Resource Management: Intelligent resource allocation and monitoring
- Performance Monitoring: Comprehensive metrics and health tracking
- Error Handling: Robust error recovery and fault tolerance

Author: Assistant
Date: 2024
Phase: Tool Integration Framework
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core framework components
try:
    from .tool_registry import (
        ToolRegistry,
        ToolMetadata,
        ToolInstance,
        ToolStatus,
        ToolCategory,
        get_tool_registry,
        register_tool,
        get_tool
    )
    
    from .agent_coordinator import (
        AgentCoordinator,
        TaskMetadata,
        AgentMetadata,
        CoordinationPlan,
        TaskStatus,
        AgentStatus,
        CoordinationStrategy,
        ResourceManager,
        TaskScheduler,
        get_agent_coordinator
    )
    
    logger.info("CrewAI Framework components loaded successfully")
    
except ImportError as e:
    logger.error(f"Failed to import framework components: {e}")
    raise

# Framework version
__version__ = "1.0.0"
__author__ = "Assistant"
__description__ = "Enhanced CrewAI Tool Integration and Agent Coordination Framework"

# Export main components
__all__ = [
    # Tool Registry
    'ToolRegistry',
    'ToolMetadata', 
    'ToolInstance',
    'ToolStatus',
    'ToolCategory',
    'get_tool_registry',
    'register_tool',
    'get_tool',
    
    # Agent Coordinator
    'AgentCoordinator',
    'TaskMetadata',
    'AgentMetadata', 
    'CoordinationPlan',
    'TaskStatus',
    'AgentStatus',
    'CoordinationStrategy',
    'ResourceManager',
    'TaskScheduler',
    'get_agent_coordinator',
    
    # Framework utilities
    'FrameworkManager',
    'initialize_framework',
    'get_framework_status',
    'shutdown_framework'
]


class FrameworkManager:
    """
    Main framework manager that coordinates all components.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the framework manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or os.path.join(
            os.path.dirname(__file__), '../config'
        )
        
        # Initialize components
        self.tool_registry: Optional[ToolRegistry] = None
        self.agent_coordinator: Optional[AgentCoordinator] = None
        self.initialized = False
        
        logger.info(f"Framework manager created with config dir: {self.config_dir}")
    
    def initialize(self) -> bool:
        """
        Initialize the framework components.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing CrewAI Framework...")
            
            # Initialize tool registry
            tool_config_path = os.path.join(self.config_dir, 'tool_registry.yaml')
            self.tool_registry = ToolRegistry(tool_config_path)
            logger.info("Tool registry initialized")
            
            # Initialize agent coordinator
            coordinator_config_path = os.path.join(self.config_dir, 'coordinator.yaml')
            self.agent_coordinator = AgentCoordinator(coordinator_config_path)
            logger.info("Agent coordinator initialized")
            
            self.initialized = True
            logger.info("CrewAI Framework initialization completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Framework initialization failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive framework status.
        
        Returns:
            Framework status dictionary
        """
        if not self.initialized:
            return {
                'initialized': False,
                'error': 'Framework not initialized'
            }
        
        try:
            status = {
                'initialized': True,
                'version': __version__,
                'components': {
                    'tool_registry': {
                        'status': 'active' if self.tool_registry else 'inactive',
                        'details': self.tool_registry.get_registry_status() if self.tool_registry else None
                    },
                    'agent_coordinator': {
                        'status': 'active' if self.agent_coordinator else 'inactive',
                        'details': self.agent_coordinator.get_coordinator_status() if self.agent_coordinator else None
                    }
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting framework status: {e}")
            return {
                'initialized': True,
                'version': __version__,
                'error': str(e),
                'components': {
                    'tool_registry': {
                        'status': 'error',
                        'details': None
                    },
                    'agent_coordinator': {
                        'status': 'error',
                        'details': None
                    }
                }
            }
    
    def register_tool_from_class(self, tool_name: str, tool_class: Type, 
                                metadata: Optional[ToolMetadata] = None,
                                config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a tool class with the framework.
        
        Args:
            tool_name: Name of the tool
            tool_class: Tool class
            metadata: Tool metadata
            config: Tool configuration
            
        Returns:
            True if registration successful
        """
        if not self.initialized or not self.tool_registry:
            logger.error("Framework not initialized")
            return False
        
        return self.tool_registry.register_tool(tool_name, tool_class, metadata, config)
    
    def get_tool_instance(self, tool_name: str, **kwargs):
        """
        Get a tool instance from the registry.
        
        Args:
            tool_name: Name of the tool
            **kwargs: Additional arguments for tool initialization
            
        Returns:
            Tool instance or None
        """
        if not self.initialized or not self.tool_registry:
            logger.error("Framework not initialized")
            return None
        
        return self.tool_registry.get_tool(tool_name, **kwargs)
    
    def register_agent(self, agent_id: str, agent, metadata: Optional[AgentMetadata] = None) -> bool:
        """
        Register an agent with the coordinator.
        
        Args:
            agent_id: Agent identifier
            agent: Agent instance
            metadata: Agent metadata
            
        Returns:
            True if registration successful
        """
        if not self.initialized or not self.agent_coordinator:
            logger.error("Framework not initialized")
            return False
        
        return self.agent_coordinator.register_agent(agent_id, agent, metadata)
    
    def create_coordination_plan(self, plan_id: str, task_ids: List[str], 
                               strategy: CoordinationStrategy = CoordinationStrategy.ADAPTIVE):
        """
        Create a coordination plan.
        
        Args:
            plan_id: Plan identifier
            task_ids: List of task IDs
            strategy: Coordination strategy
            
        Returns:
            Coordination plan or None
        """
        if not self.initialized or not self.agent_coordinator:
            logger.error("Framework not initialized")
            return None
        
        return self.agent_coordinator.create_coordination_plan(plan_id, task_ids, strategy)
    
    def execute_plan(self, plan_id: str) -> bool:
        """
        Execute a coordination plan.
        
        Args:
            plan_id: Plan identifier
            
        Returns:
            True if execution started successfully
        """
        if not self.initialized or not self.agent_coordinator:
            logger.error("Framework not initialized")
            return False
        
        return self.agent_coordinator.execute_coordination_plan(plan_id)
    
    def shutdown(self):
        """
        Shutdown the framework gracefully.
        """
        logger.info("Shutting down CrewAI Framework...")
        
        try:
            # Export final reports
            if self.tool_registry:
                registry_export_path = "/tmp/final_tool_registry_export.yaml"
                self.tool_registry.export_registry(registry_export_path)
                logger.info(f"Tool registry exported to: {registry_export_path}")
            
            if self.agent_coordinator:
                coordinator_export_path = "/tmp/final_coordination_report.yaml"
                self.agent_coordinator.export_coordination_report(coordinator_export_path)
                logger.info(f"Coordination report exported to: {coordinator_export_path}")
            
            # Reset components
            self.tool_registry = None
            self.agent_coordinator = None
            self.initialized = False
            
            logger.info("Framework shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during framework shutdown: {e}")


# Global framework manager instance
_global_framework: Optional[FrameworkManager] = None


def initialize_framework(config_dir: Optional[str] = None) -> bool:
    """
    Initialize the global framework instance.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        True if initialization successful
    """
    global _global_framework
    
    if _global_framework is not None:
        logger.warning("Framework already initialized")
        return True
    
    _global_framework = FrameworkManager(config_dir)
    return _global_framework.initialize()


def get_framework_manager() -> Optional[FrameworkManager]:
    """
    Get the global framework manager instance.
    
    Returns:
        Framework manager or None if not initialized
    """
    return _global_framework


def get_framework_status() -> Dict[str, Any]:
    """
    Get the status of the global framework.
    
    Returns:
        Framework status dictionary
    """
    if _global_framework is None:
        return {
            'initialized': False,
            'error': 'Framework not initialized'
        }
    
    return _global_framework.get_status()


def shutdown_framework():
    """
    Shutdown the global framework instance.
    """
    global _global_framework
    
    if _global_framework is not None:
        _global_framework.shutdown()
        _global_framework = None
    else:
        logger.warning("Framework not initialized, nothing to shutdown")


# Convenience functions for common operations
def quick_register_tool(tool_name: str, tool_class: Type, **config) -> bool:
    """
    Quick tool registration with the global framework.
    
    Args:
        tool_name: Name of the tool
        tool_class: Tool class
        **config: Tool configuration
        
    Returns:
        True if registration successful
    """
    if _global_framework is None:
        logger.error("Framework not initialized. Call initialize_framework() first.")
        return False
    
    return _global_framework.register_tool_from_class(tool_name, tool_class, config=config)


def quick_get_tool(tool_name: str, **kwargs):
    """
    Quick tool retrieval from the global framework.
    
    Args:
        tool_name: Name of the tool
        **kwargs: Additional arguments for tool initialization
        
    Returns:
        Tool instance or None
    """
    if _global_framework is None:
        logger.error("Framework not initialized. Call initialize_framework() first.")
        return None
    
    return _global_framework.get_tool_instance(tool_name, **kwargs)


def quick_register_agent(agent_id: str, agent, **metadata_kwargs) -> bool:
    """
    Quick agent registration with the global framework.
    
    Args:
        agent_id: Agent identifier
        agent: Agent instance
        **metadata_kwargs: Agent metadata parameters
        
    Returns:
        True if registration successful
    """
    if _global_framework is None:
        logger.error("Framework not initialized. Call initialize_framework() first.")
        return False
    
    # Create metadata if provided
    metadata = None
    if metadata_kwargs:
        metadata = AgentMetadata(
            agent_id=agent_id,
            name=metadata_kwargs.get('name', agent.role),
            role=agent.role,
            **{k: v for k, v in metadata_kwargs.items() if k != 'name'}
        )
    
    return _global_framework.register_agent(agent_id, agent, metadata)


# Framework information
def get_framework_info() -> Dict[str, Any]:
    """
    Get framework information.
    
    Returns:
        Framework information dictionary
    """
    return {
        'name': 'CrewAI Integration Framework',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'components': [
            'Tool Registry',
            'Agent Coordinator', 
            'Resource Manager',
            'Task Scheduler',
            'Performance Monitor'
        ],
        'features': [
            'Dynamic tool discovery',
            'Centralized tool management',
            'Enhanced agent coordination',
            'Intelligent resource allocation',
            'Advanced task scheduling',
            'Performance monitoring',
            'Error handling and recovery',
            'Configuration management'
        ]
    }


if __name__ == "__main__":
    # Demo usage
    print("CrewAI Integration Framework")
    print("=" * 50)
    
    # Print framework info
    info = get_framework_info()
    print(f"Name: {info['name']}")
    print(f"Version: {info['version']}")
    print(f"Description: {info['description']}")
    
    print("\nComponents:")
    for component in info['components']:
        print(f"  - {component}")
    
    print("\nFeatures:")
    for feature in info['features']:
        print(f"  - {feature}")
    
    # Initialize framework
    print("\nInitializing framework...")
    success = initialize_framework()
    
    if success:
        print("✅ Framework initialized successfully")
        
        # Get status
        status = get_framework_status()
        print(f"\nFramework Status:")
        print(f"  Initialized: {status['initialized']}")
        print(f"  Version: {status['version']}")
        
        # Component status
        for component, details in status['components'].items():
            print(f"  {component}: {details['status']}")
        
        # Shutdown
        print("\nShutting down framework...")
        shutdown_framework()
        print("✅ Framework shutdown completed")
        
    else:
        print("❌ Framework initialization failed")