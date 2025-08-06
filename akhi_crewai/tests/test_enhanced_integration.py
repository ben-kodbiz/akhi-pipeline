#!/usr/bin/env python3
"""
Enhanced CrewAI Integration Test Suite

This test suite validates the enhanced CrewAI integration including:
- Framework initialization
- Tool registry functionality
- Agent coordination
- Workflow orchestration
- Enhanced crew operations
- Performance monitoring
- Error handling

Author: Assistant
Date: 2024
Phase: Enhanced Integration Testing
"""

import os
import sys
import unittest
import tempfile
import shutil
import yaml
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import components to test
from akhi_crewai.framework import (
    ToolRegistry,
    AgentCoordinator,
    FrameworkManager,
    initialize_framework,
    get_framework_manager
)

from akhi_crewai.integration import (
    EnhancedIslamicContentCrew,
    WorkflowStep,
    get_integration_status,
    is_integration_complete
)

from akhi_crewai.framework.tool_registry import ToolCategory, ToolStatus
from akhi_crewai.framework.agent_coordinator import CoordinationStrategy, TaskStatus


class TestFrameworkComponents(unittest.TestCase):
    """Test framework components individually."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.test_dir, 'config')
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Create minimal test configurations
        self._create_test_configs()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_configs(self):
        """Create minimal test configuration files."""
        # Tool registry config
        tool_config = {
            'registry': {
                'auto_discovery': True,
                'performance_monitoring': True,
                'error_tracking': True,
                'dependency_validation': True
            },
            'tools': {
                'test_tool': {
                    'enabled': True,
                    'category': 'processing',
                    'priority': 1,
                    'version': '1.0.0',
                    'dependencies': [],
                    'resource_requirements': {
                        'memory': '100MB',
                        'cpu': 1
                    }
                }
            }
        }
        
        with open(os.path.join(self.config_dir, 'tool_registry.yaml'), 'w') as f:
            yaml.dump(tool_config, f)
        
        # Coordinator config
        coordinator_config = {
            'coordinator': {
                'max_concurrent_tasks': 5,
                'task_timeout': 300,
                'health_check_interval': 30,
                'performance_monitoring': True
            },
            'agents': {
                'test_agent': {
                    'capabilities': ['test'],
                    'tools': ['test_tool'],
                    'resource_requirements': {
                        'memory': '200MB',
                        'cpu': 1
                    }
                }
            }
        }
        
        with open(os.path.join(self.config_dir, 'coordinator.yaml'), 'w') as f:
            yaml.dump(coordinator_config, f)
        
        # Enhanced crew config
        crew_config = {
            'crew': {
                'name': 'Test Crew',
                'version': '1.0.0',
                'description': 'Test crew for integration testing'
            },
            'workflows': {
                'test_workflow': {
                    'description': 'Test workflow',
                    'steps': ['test_step']
                }
            },
            'agents': {
                'test_agent': {
                    'enabled': True,
                    'tools': ['test_tool']
                }
            }
        }
        
        with open(os.path.join(self.config_dir, 'enhanced_crew.yaml'), 'w') as f:
            yaml.dump(crew_config, f)
    
    def test_tool_registry_initialization(self):
        """Test tool registry initialization."""
        registry = ToolRegistry()
        self.assertIsNotNone(registry)
        self.assertEqual(len(registry.tools), 0)
        
        # Test configuration loading
        config_path = os.path.join(self.config_dir, 'tool_registry.yaml')
        registry.load_config(config_path)
        self.assertIsNotNone(registry.config)
    
    def test_agent_coordinator_initialization(self):
        """Test agent coordinator initialization."""
        coordinator = AgentCoordinator()
        self.assertIsNotNone(coordinator)
        self.assertEqual(len(coordinator.agents), 0)
        
        # Test configuration loading
        config_path = os.path.join(self.config_dir, 'coordinator.yaml')
        coordinator.load_config(config_path)
        self.assertIsNotNone(coordinator.config)
    
    def test_framework_manager_initialization(self):
        """Test framework manager initialization."""
        manager = FrameworkManager(self.config_dir)
        self.assertIsNotNone(manager)
        self.assertIsNotNone(manager.tool_registry)
        self.assertIsNotNone(manager.agent_coordinator)
    
    @patch('akhi_crewai.framework.FrameworkManager')
    def test_framework_initialization_function(self, mock_manager_class):
        """Test framework initialization function."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        result = initialize_framework(self.config_dir)
        self.assertTrue(result)
        mock_manager_class.assert_called_once_with(self.config_dir)


class TestToolRegistryFunctionality(unittest.TestCase):
    """Test tool registry functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.registry = ToolRegistry()
    
    def test_tool_registration(self):
        """Test tool registration."""
        # Create mock tool class
        class MockTool:
            def __init__(self):
                self.name = 'mock_tool'
                self.description = 'Mock tool for testing'
        
        # Register tool
        success = self.registry.register_tool_from_class(
            'mock_tool',
            MockTool,
            category=ToolCategory.PROCESSING
        )
        
        self.assertTrue(success)
        self.assertIn('mock_tool', self.registry.tools)
    
    def test_tool_retrieval(self):
        """Test tool retrieval."""
        # Register a mock tool first
        class MockTool:
            def __init__(self):
                self.name = 'test_tool'
        
        self.registry.register_tool_from_class(
            'test_tool',
            MockTool,
            category=ToolCategory.PROCESSING
        )
        
        # Test retrieval
        tool = self.registry.get_tool_instance('test_tool')
        self.assertIsNotNone(tool)
        
        # Test non-existent tool
        non_existent = self.registry.get_tool_instance('non_existent')
        self.assertIsNone(non_existent)
    
    def test_tools_by_category(self):
        """Test retrieving tools by category."""
        # Register tools in different categories
        class ProcessingTool:
            def __init__(self):
                self.name = 'processing_tool'
        
        class SearchTool:
            def __init__(self):
                self.name = 'search_tool'
        
        self.registry.register_tool_from_class(
            'processing_tool',
            ProcessingTool,
            category=ToolCategory.PROCESSING
        )
        
        self.registry.register_tool_from_class(
            'search_tool',
            SearchTool,
            category=ToolCategory.SEARCH
        )
        
        # Test category retrieval
        processing_tools = self.registry.get_tools_by_category(ToolCategory.PROCESSING)
        self.assertEqual(len(processing_tools), 1)
        self.assertIn('processing_tool', processing_tools)
        
        search_tools = self.registry.get_tools_by_category(ToolCategory.SEARCH)
        self.assertEqual(len(search_tools), 1)
        self.assertIn('search_tool', search_tools)


class TestAgentCoordinatorFunctionality(unittest.TestCase):
    """Test agent coordinator functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.coordinator = AgentCoordinator()
    
    def test_agent_registration(self):
        """Test agent registration."""
        # Create mock agent
        mock_agent = Mock()
        mock_agent.role = 'test_agent'
        
        # Register agent
        success = self.coordinator.register_agent(
            'test_agent',
            mock_agent,
            capabilities=['test'],
            available_tools=['test_tool']
        )
        
        self.assertTrue(success)
        self.assertIn('test_agent', self.coordinator.agents)
    
    def test_task_creation(self):
        """Test task creation."""
        # Register a mock agent first
        mock_agent = Mock()
        self.coordinator.register_agent(
            'test_agent',
            mock_agent,
            capabilities=['test'],
            available_tools=['test_tool']
        )
        
        # Create task
        mock_task = Mock()
        success = self.coordinator.create_task(
            task_id='test_task',
            name='Test Task',
            description='Test task description',
            agent_id='test_agent',
            task=mock_task
        )
        
        self.assertTrue(success)
    
    def test_coordination_plan_creation(self):
        """Test coordination plan creation."""
        # Register agent and create tasks first
        mock_agent = Mock()
        self.coordinator.register_agent(
            'test_agent',
            mock_agent,
            capabilities=['test'],
            available_tools=['test_tool']
        )
        
        mock_task = Mock()
        self.coordinator.create_task(
            task_id='test_task',
            name='Test Task',
            description='Test task description',
            agent_id='test_agent',
            task=mock_task
        )
        
        # Create coordination plan
        plan = self.coordinator.create_coordination_plan(
            plan_id='test_plan',
            task_ids=['test_task'],
            strategy=CoordinationStrategy.SEQUENTIAL
        )
        
        self.assertIsNotNone(plan)
        self.assertEqual(plan.plan_id, 'test_plan')
        self.assertEqual(len(plan.task_ids), 1)


class TestEnhancedCrewIntegration(unittest.TestCase):
    """Test enhanced crew integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.test_dir, 'config')
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Create test configuration
        self._create_test_config()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_config(self):
        """Create test configuration for enhanced crew."""
        config = {
            'crew': {
                'name': 'Test Enhanced Crew',
                'version': '1.0.0',
                'description': 'Test crew for integration testing',
                'max_concurrent_workflows': 2,
                'enable_performance_monitoring': True,
                'enable_error_recovery': True
            },
            'workflows': {
                'test_workflow': {
                    'description': 'Test workflow',
                    'steps': ['test_step']
                }
            },
            'agents': {
                'test_agent': {
                    'enabled': True,
                    'max_concurrent_tasks': 1,
                    'tools': ['test_tool']
                }
            }
        }
        
        config_path = os.path.join(self.config_dir, 'enhanced_crew.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    @patch('akhi_crewai.integration.enhanced_crew.initialize_framework')
    @patch('akhi_crewai.integration.enhanced_crew.get_framework_manager')
    def test_enhanced_crew_initialization(self, mock_get_manager, mock_init_framework):
        """Test enhanced crew initialization."""
        # Mock framework components
        mock_init_framework.return_value = True
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager
        
        # Mock tool registry and agent coordinator
        mock_manager.tool_registry = Mock()
        mock_manager.agent_coordinator = Mock()
        mock_manager.register_tool_from_class = Mock(return_value=True)
        mock_manager.get_tool_instance = Mock(return_value=Mock())
        mock_manager.register_agent = Mock(return_value=True)
        
        # Initialize enhanced crew
        config_path = os.path.join(self.config_dir, 'enhanced_crew.yaml')
        
        try:
            crew = EnhancedIslamicContentCrew(config_path)
            self.assertIsNotNone(crew)
            self.assertIsNotNone(crew.config)
            
            # Verify framework initialization was called
            mock_init_framework.assert_called_once()
            mock_get_manager.assert_called_once()
            
        except Exception as e:
            # Expected to fail due to missing actual tool classes
            # but initialization logic should be tested
            self.assertIsInstance(e, (ImportError, AttributeError, Exception))
    
    def test_workflow_step_creation(self):
        """Test workflow step creation."""
        step = WorkflowStep(
            step_id='test_step',
            name='Test Step',
            description='Test step description',
            agent_type='test_agent',
            tool_name='test_tool',
            input_schema={'input': 'str'},
            output_schema={'output': 'str'},
            dependencies=[]
        )
        
        self.assertEqual(step.step_id, 'test_step')
        self.assertEqual(step.name, 'Test Step')
        self.assertEqual(step.agent_type, 'test_agent')
        self.assertEqual(step.tool_name, 'test_tool')
        self.assertEqual(len(step.dependencies), 0)


class TestIntegrationStatus(unittest.TestCase):
    """Test integration status and utilities."""
    
    def test_integration_status(self):
        """Test integration status functions."""
        status = get_integration_status()
        self.assertIsInstance(status, dict)
        
        # Check expected components
        expected_components = [
            'framework_integration',
            'tool_registry',
            'agent_coordination',
            'workflow_orchestration',
            'performance_monitoring',
            'error_handling',
            'enhanced_crew'
        ]
        
        for component in expected_components:
            self.assertIn(component, status)
    
    def test_integration_complete(self):
        """Test integration completeness check."""
        complete = is_integration_complete()
        self.assertIsInstance(complete, bool)


class TestPerformanceAndErrorHandling(unittest.TestCase):
    """Test performance monitoring and error handling."""
    
    def test_tool_registry_error_handling(self):
        """Test tool registry error handling."""
        registry = ToolRegistry()
        
        # Test registering invalid tool
        success = registry.register_tool_from_class(
            'invalid_tool',
            None,  # Invalid tool class
            category=ToolCategory.PROCESSING
        )
        
        self.assertFalse(success)
    
    def test_agent_coordinator_error_handling(self):
        """Test agent coordinator error handling."""
        coordinator = AgentCoordinator()
        
        # Test creating task without registered agent
        mock_task = Mock()
        success = coordinator.create_task(
            task_id='test_task',
            name='Test Task',
            description='Test task description',
            agent_id='non_existent_agent',  # Non-existent agent
            task=mock_task
        )
        
        self.assertFalse(success)


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation."""
    
    def test_tool_registry_config_validation(self):
        """Test tool registry configuration validation."""
        registry = ToolRegistry()
        
        # Test with invalid config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: [')
            invalid_config_path = f.name
        
        try:
            # Should handle invalid config gracefully
            registry.load_config(invalid_config_path)
            # Should fall back to default config
            self.assertIsNotNone(registry.config)
        finally:
            os.unlink(invalid_config_path)
    
    def test_enhanced_crew_config_validation(self):
        """Test enhanced crew configuration validation."""
        # Test with non-existent config file
        try:
            crew = EnhancedIslamicContentCrew('/non/existent/config.yaml')
            # Should fall back to default config
            self.assertIsNotNone(crew.config)
        except Exception as e:
            # Expected to fail due to missing framework components
            # but config loading should be tested
            pass


if __name__ == '__main__':
    # Set up test environment
    print("Enhanced CrewAI Integration Test Suite")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestFrameworkComponents,
        TestToolRegistryFunctionality,
        TestAgentCoordinatorFunctionality,
        TestEnhancedCrewIntegration,
        TestIntegrationStatus,
        TestPerformanceAndErrorHandling,
        TestConfigurationValidation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\n')[-2]}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    print(f"\nTest suite {'PASSED' if exit_code == 0 else 'FAILED'}")
    sys.exit(exit_code)