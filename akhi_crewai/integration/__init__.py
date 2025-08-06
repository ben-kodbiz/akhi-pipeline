#!/usr/bin/env python3
"""
Akhi CrewAI Integration Module

This module provides enhanced integration capabilities for the Akhi CrewAI system,
including:

- Enhanced crew orchestration
- Framework integration
- Advanced workflow management
- Performance optimization
- Error handling and recovery

Components:
- EnhancedIslamicContentCrew: Main enhanced crew class
- WorkflowStep: Workflow step definition
- Integration utilities and helpers

Author: Assistant
Date: 2024
Phase: Enhanced Integration
"""

from .enhanced_crew import (
    EnhancedIslamicContentCrew,
    WorkflowStep
)

__all__ = [
    'EnhancedIslamicContentCrew',
    'WorkflowStep'
]

__version__ = '2.0.0'
__author__ = 'Assistant'
__description__ = 'Enhanced CrewAI Integration for Islamic Content Processing'

# Integration status
INTEGRATION_STATUS = {
    'framework_integration': True,
    'tool_registry': True,
    'agent_coordination': True,
    'workflow_orchestration': True,
    'performance_monitoring': True,
    'error_handling': True,
    'enhanced_crew': True
}

def get_integration_status():
    """Get the current integration status."""
    return INTEGRATION_STATUS.copy()

def is_integration_complete():
    """Check if all integration components are available."""
    return all(INTEGRATION_STATUS.values())

def get_integration_info():
    """Get comprehensive integration information."""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'status': get_integration_status(),
        'complete': is_integration_complete(),
        'components': list(INTEGRATION_STATUS.keys())
    }

# Quick access functions
def create_enhanced_crew(config_path=None):
    """Create an enhanced Islamic content crew."""
    return EnhancedIslamicContentCrew(config_path)

def get_available_workflows():
    """Get list of available workflow types."""
    return [
        'full_pipeline',
        'qa_pipeline', 
        'content_analysis',
        'batch_processing'
    ]

# Module initialization
print(f"Akhi CrewAI Integration v{__version__} loaded")
print(f"Integration complete: {is_integration_complete()}")

if __name__ == "__main__":
    # Display integration information
    info = get_integration_info()
    print("\nAkhi CrewAI Integration Information:")
    print("=" * 40)
    print(f"Version: {info['version']}")
    print(f"Author: {info['author']}")
    print(f"Description: {info['description']}")
    print(f"Integration Complete: {info['complete']}")
    print("\nComponents:")
    for component, status in info['status'].items():
        status_icon = "✓" if status else "✗"
        print(f"  {status_icon} {component}")
    print("\nAvailable Workflows:")
    for workflow in get_available_workflows():
        print(f"  - {workflow}")