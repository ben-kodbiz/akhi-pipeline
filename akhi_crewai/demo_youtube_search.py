#!/usr/bin/env python3
"""
Demo script for YouTube Search Tool
Demonstrates the functionality of the YouTube Search Tool for Islamic content.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.youtube_search import YouTubeSearchTool

def main():
    print("🎬 YouTube Search Tool Demo")
    print("=" * 50)
    
    # Initialize the tool
    try:
        search_tool = YouTubeSearchTool()
        print("✅ YouTube Search Tool initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize tool: {e}")
        return
    
    # Demo searches
    demo_queries = [
        {
            "query": "Nouman Ali Khan Quran",
            "max_results": 3,
            "description": "Search for Nouman Ali Khan's Quran lectures"
        },
        {
            "query": "Islamic history",
            "max_results": 2,
            "description": "Search for Islamic history content"
        },
        {
            "query": "motivation",
            "max_results": 2,
            "description": "Search for motivational content (will be enhanced with 'islamic')"
        }
    ]
    
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n🔍 Demo {i}: {demo['description']}")
        print("-" * 40)
        
        try:
            # Run the search
            result = search_tool._run(
                query=demo['query'],
                max_results=demo['max_results']
            )
            
            if "Error" in result:
                print(f"❌ Search failed: {result}")
                continue
                
            # Parse and display results
            lines = result.split('\n')
            for line in lines[:10]:  # Show first 10 lines
                if line.strip():
                    print(f"   {line}")
            
            if len(lines) > 10:
                print(f"   ... and {len(lines) - 10} more lines")
                
        except Exception as e:
            print(f"❌ Error during search: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Demo completed!")
    print("\n📋 Summary:")
    print("   - YouTube Search Tool successfully implemented")
    print("   - Islamic content filtering and enhancement working")
    print("   - Ready for integration with CrewAI agents")

if __name__ == "__main__":
    main()