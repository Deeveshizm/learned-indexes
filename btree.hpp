#pragma once
#include <vector>
#include <algorithm>

// Simple B-Tree implementation for comparison
template<typename Key, typename Value, size_t PAGE_SIZE = 128>
class BTree {
private:
    struct Node {
        std::vector<Key> keys;
        std::vector<Value> values;
        std::vector<Node*> children;
        bool is_leaf;
        
        Node(bool leaf = true) : is_leaf(leaf) {
            keys.reserve(PAGE_SIZE);
            if (leaf) {
                values.reserve(PAGE_SIZE);
            } else {
                children.reserve(PAGE_SIZE + 1);
            }
        }
        
        ~Node() {
            for (auto child : children) {
                delete child;
            }
        }
    };
    
    Node* root;
    size_t total_size;
    
public:
    BTree() : root(new Node(true)), total_size(0) {}
    
    ~BTree() {
        delete root;
    }
    
    // Build from sorted data (bulk loading)
    void build(const std::vector<std::pair<Key, Value>>& sorted_data) {
        delete root;
        root = new Node(true);
        total_size = 0;
        
        if (sorted_data.empty()) return;
        
        // Simple bulk load - just put everything in leaf nodes
        std::vector<Node*> leaves;
        Node* current_leaf = new Node(true);
        
        for (const auto& [key, value] : sorted_data) {
            if (current_leaf->keys.size() >= PAGE_SIZE) {
                leaves.push_back(current_leaf);
                current_leaf = new Node(true);
            }
            current_leaf->keys.push_back(key);
            current_leaf->values.push_back(value);
            total_size++;
        }
        
        if (!current_leaf->keys.empty()) {
            leaves.push_back(current_leaf);
        }
        
        // Build tree bottom-up
        while (leaves.size() > 1) {
            std::vector<Node*> parents;
            Node* parent = new Node(false);
            
            for (auto leaf : leaves) {
                if (parent->children.size() >= PAGE_SIZE) {
                    parents.push_back(parent);
                    parent = new Node(false);
                }
                
                if (!parent->children.empty()) {
                    parent->keys.push_back(leaf->keys.front());
                }
                parent->children.push_back(leaf);
            }
            
            if (!parent->children.empty()) {
                parents.push_back(parent);
            }
            
            leaves = parents;
        }
        
        if (!leaves.empty()) {
            root = leaves[0];
        }
    }
    
    // Lookup
    Value* find(const Key& key) {
        return find_in_node(root, key);
    }
    
    // Lower bound (first key >= search key)
    size_t lower_bound(const Key& key) {
        return lower_bound_in_node(root, key, 0);
    }
    
    size_t get_size_bytes() const {
        return calculate_size(root);
    }
    
private:
    Value* find_in_node(Node* node, const Key& key) {
        if (!node) return nullptr;
        
        auto it = std::lower_bound(node->keys.begin(), node->keys.end(), key);
        
        if (node->is_leaf) {
            if (it != node->keys.end() && *it == key) {
                size_t idx = it - node->keys.begin();
                return &node->values[idx];
            }
            return nullptr;
        } else {
            size_t idx = it - node->keys.begin();
            if (idx >= node->children.size()) {
                idx = node->children.size() - 1;
            }
            return find_in_node(node->children[idx], key);
        }
    }
    
    size_t lower_bound_in_node(Node* node, const Key& key, size_t offset) {
        if (!node) return offset;
        
        auto it = std::lower_bound(node->keys.begin(), node->keys.end(), key);
        
        if (node->is_leaf) {
            size_t idx = it - node->keys.begin();
            return offset + idx;
        } else {
            size_t idx = it - node->keys.begin();
            if (idx >= node->children.size()) {
                idx = node->children.size() - 1;
            }
            
            // Calculate offset for this child
            size_t child_offset = offset;
            for (size_t i = 0; i < idx; ++i) {
                child_offset += count_keys(node->children[i]);
            }
            
            return lower_bound_in_node(node->children[idx], key, child_offset);
        }
    }
    
    size_t count_keys(Node* node) const {
        if (!node) return 0;
        
        if (node->is_leaf) {
            return node->keys.size();
        }
        
        size_t count = 0;
        for (auto child : node->children) {
            count += count_keys(child);
        }
        return count;
    }
    
    size_t calculate_size(Node* node) const {
        if (!node) return 0;
        
        size_t size = sizeof(Node);
        size += node->keys.capacity() * sizeof(Key);
        size += node->values.capacity() * sizeof(Value);
        size += node->children.capacity() * sizeof(Node*);
        
        for (auto child : node->children) {
            size += calculate_size(child);
        }
        
        return size;
    }
};
