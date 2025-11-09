import { useState, useMemo } from 'react';
import { Search, ChevronRight, ChevronDown } from 'lucide-react';
import { TEST_HIERARCHY, getAllTests } from '../data/testHierarchy';
import type { TestGroup } from '../types/benchmark';

interface Props {
  selectedTestId: string | null;
  onSelectTest: (testId: string) => void;
}

export default function BenchmarkSidebar({ selectedTestId, onSelectTest }: Props) {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set(['matrix-ops', 'elementwise']));
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set());

  // Filter tests based on search query
  const filteredHierarchy = useMemo(() => {
    if (!searchQuery.trim()) {
      return TEST_HIERARCHY;
    }

    const query = searchQuery.toLowerCase();
    const allTests = getAllTests();

    // Find matching tests
    const matchingTests = allTests.filter(({ test, categoryName, groupName }) => {
      return (
        test.name.toLowerCase().includes(query) ||
        test.description.toLowerCase().includes(query) ||
        test.operation.toLowerCase().includes(query) ||
        categoryName.toLowerCase().includes(query) ||
        groupName.toLowerCase().includes(query)
      );
    });

    // Rebuild hierarchy with only matching tests
    const filteredGroups: TestGroup[] = [];
    const groupMap = new Map<string, TestGroup>();

    for (const { groupId, categoryId, test } of matchingTests) {
      const originalGroup = TEST_HIERARCHY.find(g => g.id === groupId);
      const originalCategory = originalGroup?.categories.find(c => c.id === categoryId);

      if (!originalGroup || !originalCategory) continue;

      // Get or create group
      let group = groupMap.get(groupId);
      if (!group) {
        group = {
          ...originalGroup,
          categories: [],
        };
        groupMap.set(groupId, group);
        filteredGroups.push(group);
      }

      // Find or create category in filtered group
      let category = group.categories.find(c => c.id === categoryId);
      if (!category) {
        category = {
          ...originalCategory,
          tests: [],
        };
        group.categories.push(category);
      }

      // Add test to category
      category.tests.push(test);
    }

    return filteredGroups;
  }, [searchQuery]);

  // Auto-expand groups/categories when searching
  useMemo(() => {
    if (searchQuery.trim()) {
      const allGroupIds = new Set(filteredHierarchy.map(g => g.id));
      const allCategoryIds = new Set(
        filteredHierarchy.flatMap(g => g.categories.map(c => `${g.id}:${c.id}`))
      );
      setExpandedGroups(allGroupIds);
      setExpandedCategories(allCategoryIds);
    }
  }, [searchQuery, filteredHierarchy]);

  function toggleGroup(groupId: string) {
    setExpandedGroups(prev => {
      const next = new Set(prev);
      if (next.has(groupId)) {
        next.delete(groupId);
      } else {
        next.add(groupId);
      }
      return next;
    });
  }

  function toggleCategory(groupId: string, categoryId: string) {
    const key = `${groupId}:${categoryId}`;
    setExpandedCategories(prev => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }

  return (
    <div className="benchmark-sidebar">
      <div className="sidebar-header">
        <h2>Benchmarks</h2>
        <div className="search-box">
          <Search size={16} />
          <input
            type="text"
            placeholder="Search tests..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      <div className="sidebar-content">
        {filteredHierarchy.length === 0 ? (
          <div className="no-results">
            No tests found matching "{searchQuery}"
          </div>
        ) : (
          filteredHierarchy.map(group => (
            <div key={group.id} className="test-group">
              <div
                className="group-header"
                onClick={() => toggleGroup(group.id)}
              >
                {expandedGroups.has(group.id) ? (
                  <ChevronDown size={16} />
                ) : (
                  <ChevronRight size={16} />
                )}
                <span className="group-icon">{group.icon}</span>
                <span className="group-name">{group.name}</span>
              </div>

              {expandedGroups.has(group.id) && (
                <div className="group-content">
                  {group.categories.map(category => (
                    <div key={category.id} className="test-category">
                      <div
                        className="category-header"
                        onClick={() => toggleCategory(group.id, category.id)}
                      >
                        {expandedCategories.has(`${group.id}:${category.id}`) ? (
                          <ChevronDown size={14} />
                        ) : (
                          <ChevronRight size={14} />
                        )}
                        <span className="category-name">{category.name}</span>
                      </div>

                      {expandedCategories.has(`${group.id}:${category.id}`) && (
                        <div className="category-tests">
                          {category.tests.map(test => (
                            <div
                              key={test.id}
                              className={`test-item ${selectedTestId === test.id ? 'selected' : ''}`}
                              onClick={() => onSelectTest(test.id)}
                              title={test.description}
                            >
                              <span className="test-name">{test.name}</span>
                              {test.minGpuSize && (
                                <span className="gpu-hint" title="GPU beneficial above this size">
                                  ⚡
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
