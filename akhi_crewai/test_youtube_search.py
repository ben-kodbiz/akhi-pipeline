#!/usr/bin/env python3
"""Test script for YouTube Search Tool

This script tests the YouTube Search Tool implementation to ensure it works correctly
before integrating it into the CrewAI system.
"""

import os
import sys
import traceback
from typing import Dict, Any

# Add the tools directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

try:
    from youtube_search import YouTubeSearchTool, YouTubeSearchInput
except ImportError as e:
    print(f"Error importing YouTube Search Tool: {e}")
    sys.exit(1)


def test_tool_initialization():
    """Test tool initialization."""
    print("\n=== Testing Tool Initialization ===")
    try:
        tool = YouTubeSearchTool()
        print("✅ Tool initialized successfully")
        print(f"Tool name: {tool.name}")
        print(f"Tool description: {tool.description[:100]}...")
        return tool
    except Exception as e:
        print(f"❌ Tool initialization failed: {e}")
        traceback.print_exc()
        return None


def test_config_loading(tool):
    """Test configuration loading."""
    print("\n=== Testing Configuration Loading ===")
    try:
        config = tool.config
        print("✅ Configuration loaded successfully")
        
        # Check key configuration sections
        if 'youtube' in config:
            print("✅ YouTube configuration found")
        if 'islamic_content' in config:
            print("✅ Islamic content configuration found")
            scholars = config['islamic_content'].get('preferred_scholars', [])
            print(f"   Preferred scholars: {len(scholars)} configured")
        
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def test_input_schema():
    """Test input schema validation."""
    print("\n=== Testing Input Schema ===")
    try:
        # Test valid input
        valid_input = YouTubeSearchInput(
            query="Islamic lectures",
            max_results=5,
            duration="medium"
        )
        print("✅ Valid input schema works")
        print(f"   Query: {valid_input.query}")
        print(f"   Max results: {valid_input.max_results}")
        print(f"   Duration: {valid_input.duration}")
        
        # Test minimal input
        minimal_input = YouTubeSearchInput(query="Quran")
        print("✅ Minimal input schema works")
        print(f"   Query: {minimal_input.query}")
        print(f"   Max results (default): {minimal_input.max_results}")
        
        return True
    except Exception as e:
        print(f"❌ Input schema validation failed: {e}")
        return False


def test_query_enhancement(tool):
    """Test query enhancement for Islamic content."""
    print("\n=== Testing Query Enhancement ===")
    try:
        # Test queries that should be enhanced
        test_queries = [
            "leadership lessons",
            "motivation",
            "life advice",
            "Quran recitation",  # Already Islamic
            "Islamic history"     # Already Islamic
        ]
        
        for query in test_queries:
            enhanced = tool._enhance_query_for_islamic_content(query)
            print(f"   '{query}' -> '{enhanced}'")
        
        print("✅ Query enhancement working")
        return True
    except Exception as e:
        print(f"❌ Query enhancement failed: {e}")
        return False


def test_search_functionality(tool):
    """Test the actual search functionality."""
    print("\n=== Testing Search Functionality ===")
    
    # Test queries with different complexity
    test_queries = [
        {
            "query": "Nouman Ali Khan",
            "max_results": 3,
            "description": "Simple scholar search"
        },
        {
            "query": "Quran tafseer",
            "max_results": 2,
            "description": "Topic-based search"
        }
    ]
    
    for test_case in test_queries:
        print(f"\n--- Testing: {test_case['description']} ---")
        try:
            result = tool._run(
                query=test_case["query"],
                max_results=test_case["max_results"]
            )
            
            if "Error" in result:
                print(f"⚠️  Search returned error: {result[:200]}...")
            elif "No videos found" in result:
                print(f"⚠️  No videos found for query: {test_case['query']}")
            else:
                print(f"✅ Search successful for '{test_case['query']}'")
                # Print first few lines of result
                lines = result.split('\n')[:5]
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
                print("   ...")
            
        except Exception as e:
            print(f"❌ Search failed for '{test_case['query']}': {e}")
            traceback.print_exc()


def test_islamic_content_filtering(tool):
    """Test Islamic content filtering functionality."""
    print("\n=== Testing Islamic Content Filtering ===")
    try:
        # Create mock video results for testing
        mock_results = [
            {
                "title": "Nouman Ali Khan - Understanding Quran",
                "channel": "Bayyinah Institute",
                "description": "Islamic lecture about Quran interpretation",
                "url": "https://youtube.com/watch?v=test1"
            },
            {
                "title": "Random Motivational Video",
                "channel": "Random Channel",
                "description": "General motivation content",
                "url": "https://youtube.com/watch?v=test2"
            },
            {
                "title": "Yasir Qadhi - Islamic History",
                "channel": "Epic Masjid",
                "description": "Detailed Islamic history lesson",
                "url": "https://youtube.com/watch?v=test3"
            }
        ]
        
        filtered_results = tool._filter_islamic_content(mock_results)
        
        print("✅ Islamic content filtering working")
        print("   Filtered results (by Islamic relevance):")
        for i, video in enumerate(filtered_results, 1):
            score = video.get('islamic_relevance_score', 0)
            title = video.get('title', 'Unknown')
            print(f"   {i}. {title} (Score: {score})")
        
        return True
    except Exception as e:
        print(f"❌ Islamic content filtering failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting YouTube Search Tool Tests")
    print("=" * 50)
    
    # Initialize tool
    tool = test_tool_initialization()
    if not tool:
        print("\n❌ Cannot proceed with tests - tool initialization failed")
        return False
    
    # Run tests
    tests = [
        (test_config_loading, tool),
        (test_input_schema,),
        (test_query_enhancement, tool),
        (test_islamic_content_filtering, tool),
        (test_search_functionality, tool)  # This one last as it requires internet
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func, *args in tests:
        try:
            if test_func(*args):
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! YouTube Search Tool is ready.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)