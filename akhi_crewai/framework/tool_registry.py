#!/usr/bin/env python3
"""
Tool Registry and Integration Framework for CrewAI

This module provides a centralized registry for all CrewAI tools with:
- Dynamic tool discovery and registration
- Tool dependency management
- Configuration validation
- Performance monitoring
- Error handling and recovery
- Tool versioning and compatibility

Author: Assistant
Date: 2024
Phase: Tool Integration Framework
"""

import os
import sys
import json
import yaml
import logging
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Import unified config loader
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from utils.config_loader import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolStatus(Enum):
    """Tool status enumeration."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"


class ToolCategory(Enum):
    """Tool category enumeration."""
    SEARCH = "search"
    DOWNLOAD = "download"
    TRANSCRIPTION = "transcription"
    PROCESSING = "processing"
    EMBEDDING = "embedding"
    STORAGE = "storage"
    QUERY = "query"
    GENERATION = "generation"
    TRAINING = "training"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    UTILITY = "utility"


@dataclass
class ToolMetadata:
    """Metadata for registered tools."""
    name: str
    description: str
    category: ToolCategory
    version: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    status: ToolStatus = ToolStatus.AVAILABLE
    error_message: Optional[str] = None


@dataclass
class ToolInstance:
    """Represents an instantiated tool."""
    tool_class: Type[BaseTool]
    instance: Optional[BaseTool] = None
    metadata: Optional[ToolMetadata] = None
    config: Optional[Dict[str, Any]] = None
    initialization_time: Optional[float] = None
    last_used: Optional[str] = None
    usage_count: int = 0
    error_count: int = 0


class ToolRegistry:
    """
    Centralized registry for CrewAI tools with advanced management capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the tool registry.
        
        Args:
            config_path: Path to the registry configuration file
        """
        self.config = self._load_config(config_path)
        self.tools: Dict[str, ToolInstance] = {}
        self.categories: Dict[ToolCategory, List[str]] = {cat: [] for cat in ToolCategory}
        self.dependencies: Dict[str, List[str]] = {}
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Initialize registry
        self._initialize_registry()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load registry configuration from unified config.
        
        Args:
            config_path: Path to configuration file (deprecated, uses unified config)
            
        Returns:
            Configuration dictionary
        """
        try:
            config_loader = get_config()
            tool_registry_config = config_loader.get_section('tool_registry', {})
            
            # If tool_registry section doesn't exist, try to get it from the old location
            if not tool_registry_config:
                # Fallback to old config file for backward compatibility
                if config_path is None:
                    config_path = os.path.join(
                        os.path.dirname(__file__), 
                        '../config/tool_registry.yaml'
                    )
                
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    return config
                except FileNotFoundError:
                    logger.warning(f"Config file not found: {config_path}. Using defaults.")
                    return self._get_default_config()
            
            return tool_registry_config
        except Exception as e:
            logger.warning(f"Could not load tool registry config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default registry configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'registry': {
                'auto_discover': True,
                'tools_directory': '../tools',
                'cache_enabled': True,
                'performance_monitoring': True,
                'error_tracking': True,
                'dependency_validation': True
            },
            'tools': {
                'youtube_search': {
                    'enabled': True,
                    'category': 'search',
                    'priority': 1,
                    'config': {}
                },
                'youtube_downloader': {
                    'enabled': True,
                    'category': 'download',
                    'priority': 2,
                    'dependencies': ['youtube_search'],
                    'config': {}
                },
                'transcriber': {
                    'enabled': True,
                    'category': 'transcription',
                    'priority': 3,
                    'dependencies': ['youtube_downloader'],
                    'config': {}
                },
                'chunker': {
                    'enabled': True,
                    'category': 'processing',
                    'priority': 4,
                    'dependencies': ['transcriber'],
                    'config': {}
                },
                'embedder': {
                    'enabled': True,
                    'category': 'embedding',
                    'priority': 5,
                    'dependencies': ['chunker'],
                    'config': {}
                },
                'faiss_store': {
                    'enabled': True,
                    'category': 'storage',
                    'priority': 6,
                    'dependencies': ['embedder'],
                    'config': {}
                },
                'faiss_query': {
                    'enabled': True,
                    'category': 'query',
                    'priority': 7,
                    'dependencies': ['faiss_store'],
                    'config': {}
                },
                'summarizer': {
                    'enabled': True,
                    'category': 'generation',
                    'priority': 8,
                    'dependencies': ['faiss_query'],
                    'config': {}
                },
                'answer_generator': {
                    'enabled': True,
                    'category': 'generation',
                    'priority': 9,
                    'dependencies': ['faiss_query'],
                    'config': {}
                }
            }
        }
    
    def _initialize_registry(self):
        """
        Initialize the tool registry by discovering and registering tools.
        """
        logger.info("Initializing tool registry...")
        
        if self.config['registry']['auto_discover']:
            self._auto_discover_tools()
        
        # Register configured tools
        for tool_name, tool_config in self.config['tools'].items():
            if tool_config.get('enabled', True):
                self._register_tool_from_config(tool_name, tool_config)
        
        logger.info(f"Registry initialized with {len(self.tools)} tools")
    
    def _auto_discover_tools(self):
        """
        Automatically discover tools in the tools directory.
        """
        tools_dir = Path(os.path.dirname(__file__)) / self.config['registry']['tools_directory']
        
        if not tools_dir.exists():
            logger.warning(f"Tools directory not found: {tools_dir}")
            return
        
        logger.info(f"Auto-discovering tools in: {tools_dir}")
        
        for tool_file in tools_dir.glob('*.py'):
            if tool_file.name.startswith('__') or tool_file.name.startswith('demo_'):
                continue
            
            try:
                self._discover_tool_in_file(tool_file)
            except Exception as e:
                logger.error(f"Error discovering tool in {tool_file}: {e}")
    
    def _discover_tool_in_file(self, tool_file: Path):
        """
        Discover tools in a specific file.
        
        Args:
            tool_file: Path to the tool file
        """
        module_name = tool_file.stem
        spec = importlib.util.spec_from_file_location(module_name, tool_file)
        module = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error(f"Error loading module {module_name}: {e}")
            return
        
        # Find BaseTool subclasses
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseTool) and 
                obj != BaseTool and
                name.endswith('Tool')):
                
                tool_name = self._extract_tool_name(name)
                if tool_name not in self.tools:
                    self._register_discovered_tool(tool_name, obj, module_name)
    
    def _extract_tool_name(self, class_name: str) -> str:
        """
        Extract tool name from class name.
        
        Args:
            class_name: Tool class name
            
        Returns:
            Extracted tool name
        """
        # Convert CamelCase to snake_case and remove 'Tool' suffix
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        return name.replace('_tool', '')
    
    def _register_discovered_tool(self, tool_name: str, tool_class: Type[BaseTool], module_name: str):
        """
        Register a discovered tool.
        
        Args:
            tool_name: Name of the tool
            tool_class: Tool class
            module_name: Module name
        """
        try:
            # Extract metadata from tool class
            metadata = self._extract_tool_metadata(tool_class, module_name)
            
            # Create tool instance
            tool_instance = ToolInstance(
                tool_class=tool_class,
                metadata=metadata
            )
            
            self.tools[tool_name] = tool_instance
            self.categories[metadata.category].append(tool_name)
            
            logger.info(f"Discovered and registered tool: {tool_name}")
            
        except Exception as e:
            logger.error(f"Error registering discovered tool {tool_name}: {e}")
    
    def _extract_tool_metadata(self, tool_class: Type[BaseTool], module_name: str) -> ToolMetadata:
        """
        Extract metadata from a tool class.
        
        Args:
            tool_class: Tool class
            module_name: Module name
            
        Returns:
            Tool metadata
        """
        # Try to get metadata from class attributes
        name = getattr(tool_class, 'name', tool_class.__name__)
        description = getattr(tool_class, 'description', tool_class.__doc__ or '')
        
        # Determine category based on tool name/description
        category = self._determine_tool_category(name, description)
        
        return ToolMetadata(
            name=name,
            description=description,
            category=category,
            version='1.0.0',
            author='Auto-discovered'
        )
    
    def _determine_tool_category(self, name: str, description: str) -> ToolCategory:
        """
        Determine tool category based on name and description.
        
        Args:
            name: Tool name
            description: Tool description
            
        Returns:
            Tool category
        """
        name_lower = name.lower()
        desc_lower = description.lower()
        
        if 'search' in name_lower or 'search' in desc_lower:
            return ToolCategory.SEARCH
        elif 'download' in name_lower or 'download' in desc_lower:
            return ToolCategory.DOWNLOAD
        elif 'transcrib' in name_lower or 'transcrib' in desc_lower:
            return ToolCategory.TRANSCRIPTION
        elif 'chunk' in name_lower or 'chunk' in desc_lower:
            return ToolCategory.PROCESSING
        elif 'embed' in name_lower or 'embed' in desc_lower:
            return ToolCategory.EMBEDDING
        elif 'store' in name_lower or 'storage' in desc_lower:
            return ToolCategory.STORAGE
        elif 'query' in name_lower or 'query' in desc_lower:
            return ToolCategory.QUERY
        elif 'summar' in name_lower or 'answer' in name_lower or 'generat' in desc_lower:
            return ToolCategory.GENERATION
        elif 'train' in name_lower or 'train' in desc_lower:
            return ToolCategory.TRAINING
        elif 'valid' in name_lower or 'valid' in desc_lower:
            return ToolCategory.VALIDATION
        elif 'deploy' in name_lower or 'deploy' in desc_lower:
            return ToolCategory.DEPLOYMENT
        else:
            return ToolCategory.UTILITY
    
    def _register_tool_from_config(self, tool_name: str, tool_config: Dict[str, Any]):
        """
        Register a tool from configuration.
        
        Args:
            tool_name: Name of the tool
            tool_config: Tool configuration
        """
        if tool_name in self.tools:
            # Update existing tool configuration
            self.tools[tool_name].config = tool_config.get('config', {})
            if 'dependencies' in tool_config:
                self.dependencies[tool_name] = tool_config['dependencies']
        else:
            logger.warning(f"Tool {tool_name} configured but not discovered")
    
    def register_tool(self, tool_name: str, tool_class: Type[BaseTool], 
                     metadata: Optional[ToolMetadata] = None, 
                     config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Manually register a tool.
        
        Args:
            tool_name: Name of the tool
            tool_class: Tool class
            metadata: Tool metadata
            config: Tool configuration
            
        Returns:
            True if registration successful
        """
        try:
            if metadata is None:
                metadata = self._extract_tool_metadata(tool_class, 'manual')
            
            tool_instance = ToolInstance(
                tool_class=tool_class,
                metadata=metadata,
                config=config or {}
            )
            
            self.tools[tool_name] = tool_instance
            self.categories[metadata.category].append(tool_name)
            
            logger.info(f"Manually registered tool: {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering tool {tool_name}: {e}")
            return False
    
    def register_tool_from_class(self, name: str, tool_class: Type[BaseTool], 
                                category: ToolCategory = ToolCategory.UTILITY,
                                metadata: Optional[ToolMetadata] = None) -> bool:
        """
        Register a tool from a class with optional metadata.
        
        Args:
            name: Tool name
            tool_class: Tool class
            category: Tool category
            metadata: Optional tool metadata
            
        Returns:
            True if registration successful
        """
        try:
            if metadata is None:
                metadata = ToolMetadata(
                    name=name,
                    description=getattr(tool_class, 'description', f'{name} tool'),
                    category=category,
                    version='1.0.0',
                    author='System'
                )
            
            return self.register_tool(name, tool_class, metadata)
            
        except Exception as e:
            logger.error(f"Failed to register tool from class {name}: {e}")
            return False
    
    def get_tool(self, tool_name: str, **kwargs) -> Optional[BaseTool]:
        """
        Get an instantiated tool.
        
        Args:
            tool_name: Name of the tool
            **kwargs: Additional arguments for tool initialization
            
        Returns:
            Tool instance or None if not found
        """
        if tool_name not in self.tools:
            logger.error(f"Tool not found: {tool_name}")
            return None
        
        tool_instance = self.tools[tool_name]
        
        # Check if tool is already instantiated
        if tool_instance.instance is None:
            try:
                # Merge config with kwargs
                init_config = {**(tool_instance.config or {}), **kwargs}
                
                # Instantiate tool
                start_time = datetime.now()
                tool_instance.instance = tool_instance.tool_class(**init_config)
                tool_instance.initialization_time = (datetime.now() - start_time).total_seconds()
                
                logger.info(f"Instantiated tool: {tool_name}")
                
            except Exception as e:
                logger.error(f"Error instantiating tool {tool_name}: {e}")
                tool_instance.metadata.status = ToolStatus.ERROR
                tool_instance.metadata.error_message = str(e)
                return None
        
        # Update usage statistics
        tool_instance.usage_count += 1
        tool_instance.last_used = datetime.now().isoformat()
        
        return tool_instance.instance
    
    def get_tool_instance(self, tool_name: str) -> Optional[ToolInstance]:
        """
        Get a tool instance by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ToolInstance or None if not found
        """
        return self.tools.get(tool_name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[str]:
        """
        Get tools by category.
        
        Args:
            category: Tool category
            
        Returns:
            List of tool names in the category
        """
        return self.categories.get(category, [])
    
    def get_tool_dependencies(self, tool_name: str) -> List[str]:
        """
        Get tool dependencies.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            List of dependency tool names
        """
        return self.dependencies.get(tool_name, [])
    
    def validate_dependencies(self, tool_name: str) -> bool:
        """
        Validate that all tool dependencies are available.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if all dependencies are available
        """
        dependencies = self.get_tool_dependencies(tool_name)
        
        for dep in dependencies:
            if dep not in self.tools:
                logger.error(f"Missing dependency {dep} for tool {tool_name}")
                return False
            
            if self.tools[dep].metadata.status != ToolStatus.AVAILABLE:
                logger.error(f"Dependency {dep} not available for tool {tool_name}")
                return False
        
        return True
    
    def get_tool_chain(self, start_tool: str, end_tool: str) -> Optional[List[str]]:
        """
        Get a tool chain from start to end tool.
        
        Args:
            start_tool: Starting tool name
            end_tool: Ending tool name
            
        Returns:
            List of tool names in execution order or None if no path found
        """
        # Simple topological sort based on dependencies
        visited = set()
        path = []
        
        def dfs(tool: str, target: str) -> bool:
            if tool == target:
                path.append(tool)
                return True
            
            if tool in visited:
                return False
            
            visited.add(tool)
            
            # Check tools that depend on this tool
            for name, deps in self.dependencies.items():
                if tool in deps and name not in visited:
                    if dfs(name, target):
                        path.append(tool)
                        return True
            
            return False
        
        if dfs(start_tool, end_tool):
            return list(reversed(path))
        
        return None
    
    def get_registry_status(self) -> Dict[str, Any]:
        """
        Get registry status and statistics.
        
        Returns:
            Registry status dictionary
        """
        status = {
            'total_tools': len(self.tools),
            'available_tools': sum(1 for t in self.tools.values() 
                                 if t.metadata.status == ToolStatus.AVAILABLE),
            'categories': {cat.value: len(tool_list) for cat, tool_list in self.categories.items()},
            'tools': {}
        }
        
        for name, tool in self.tools.items():
            status['tools'][name] = {
                'status': tool.metadata.status.value,
                'category': tool.metadata.category.value,
                'usage_count': tool.usage_count,
                'error_count': tool.error_count,
                'last_used': tool.last_used,
                'instantiated': tool.instance is not None
            }
        
        return status
    
    def export_registry(self, output_path: str):
        """
        Export registry configuration to file.
        
        Args:
            output_path: Output file path
        """
        registry_data = {
            'registry_info': {
                'exported_at': datetime.now().isoformat(),
                'total_tools': len(self.tools)
            },
            'tools': {}
        }
        
        for name, tool in self.tools.items():
            registry_data['tools'][name] = {
                'metadata': {
                    'name': tool.metadata.name,
                    'description': tool.metadata.description,
                    'category': tool.metadata.category.value,
                    'version': tool.metadata.version,
                    'status': tool.metadata.status.value
                },
                'config': tool.config,
                'dependencies': self.dependencies.get(name, []),
                'statistics': {
                    'usage_count': tool.usage_count,
                    'error_count': tool.error_count,
                    'last_used': tool.last_used
                }
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(registry_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Registry exported to: {output_path}")


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.
    
    Returns:
        Global tool registry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(tool_name: str, tool_class: Type[BaseTool], 
                 metadata: Optional[ToolMetadata] = None, 
                 config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Register a tool with the global registry.
    
    Args:
        tool_name: Name of the tool
        tool_class: Tool class
        metadata: Tool metadata
        config: Tool configuration
        
    Returns:
        True if registration successful
    """
    return get_tool_registry().register_tool(tool_name, tool_class, metadata, config)


def get_tool(tool_name: str, **kwargs) -> Optional[BaseTool]:
    """
    Get a tool from the global registry.
    
    Args:
        tool_name: Name of the tool
        **kwargs: Additional arguments for tool initialization
        
    Returns:
        Tool instance or None if not found
    """
    return get_tool_registry().get_tool(tool_name, **kwargs)


if __name__ == "__main__":
    # Demo usage
    print("Tool Registry Demo")
    print("=" * 50)
    
    # Initialize registry
    registry = ToolRegistry()
    
    # Print status
    status = registry.get_registry_status()
    print(f"Total tools: {status['total_tools']}")
    print(f"Available tools: {status['available_tools']}")
    
    print("\nTools by category:")
    for category, count in status['categories'].items():
        if count > 0:
            print(f"  {category}: {count}")
    
    print("\nRegistered tools:")
    for name, info in status['tools'].items():
        print(f"  {name}: {info['status']} ({info['category']})")
    
    # Test tool retrieval
    print("\nTesting tool retrieval:")
    youtube_search = registry.get_tool('youtube_search')
    if youtube_search:
        print(f"✅ Successfully retrieved: {youtube_search.__class__.__name__}")
    else:
        print("❌ Failed to retrieve youtube_search tool")
    
    # Export registry
    output_path = "/tmp/tool_registry_export.yaml"
    registry.export_registry(output_path)
    print(f"\nRegistry exported to: {output_path}")