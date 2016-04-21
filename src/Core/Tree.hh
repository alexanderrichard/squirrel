/*
 * Copyright 2016 Alexander Richard
 *
 * This file is part of Squirrel.
 *
 * Licensed under the Academic Free License 3.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You should have received a copy of the License along with Squirrel.
 * If not, see <https://opensource.org/licenses/AFL-3.0>.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifndef CORE_TREE_HH_
#define CORE_TREE_HH_

#include <Core/CommonHeaders.hh>

namespace Core {

/*
 * templated tree structure
 */
template<typename K, typename V>
class Tree
{
public:
	typedef K Key;
	typedef V Value;
	typedef std::vector<Key> Path;

protected:
	/*** Node ***/
	class Node
	{
	private:
		std::vector<Node> children_;
		Key key_;
		Value value_;
	public:
		Node(const Key& key, const Value& value);
		u32 nChildren() const;
		const Key& key() const;
		Key& key();
		const Value& value() const;
		Value& value();
		const Node& child(u32 index) const;
		Node& child(u32 index);
		bool isLeaf() const;
		/*
		 * @return true if the child could be added, false if a child with key already exists
		 */
		bool addChild(const Key& key, const Value& value);
	};
	/*** End Node ***/

protected:
	Node root_;
	void revertPath(const Path& sourcePath, Path& targetPath) const;
	bool pathExists(Path& path, const Node& node) const;
	bool keyExists(const Key& key, const Node& node) const;
	bool addNode(Path& path, const Key& key, const Value& value, Node& node);
	bool addPath(Path& path, const Value& defaultValue, Node& node);
	bool setValue(Path& path, const Value& value, Node& node);
	const Value& value(Path& path, const Node& node) const;
	Value& value(Path& path, Node& node);
	const Node& node(const Path& path) const;
	Node& node(const Path& path);
public:
	Tree(const Key& rootKey, const Value& rootValue);
	/*
	 * @return true if key exists in the tree
	 */
	bool keyExists(const Key& key) const;
	/*
	 * @return true if path exists
	 */
	bool pathExists(const Path& path) const;
	/*
	 * @return true if (key, node) could be added at the end of path, else false (e.g. if path does not exist or node already exists)
	 */
	bool addNode(const Path& path, const Key& key, const Value& value);
	/*
	 * add the given path to the tree and assign the default value to each new node along the path
	 * @return true if path could be added, else false (e.g. if path does not start with the root node)
	 */
	bool addPath(const Path& path, const Value& defaultValue);
	/*
	 * @return true if value of the node at the end of path could be set to value, else false (e.g. if path does not exists)
	 */
	bool setValue(const Path& path, const Value& value);
	/*
	 * @return value of the node at the end of path
	 * aborts with an error message if path does not exist
	 */
	const Value& value(const Path& path) const;
	/*
	 * @return value of the node at the end of path
	 * aborts with an error message if path does not exist
	 */
	Value& value(const Path& path);
};

/*** Node ***/
template<typename K, typename V>
Tree<K,V>::Node::Node(const Key& key, const Value& value) :
		key_(key),
		value_(value)
{}

template<typename K, typename V>
u32 Tree<K,V>::Node::nChildren() const {
	return children_.size();
}

template<typename K, typename V>
bool Tree<K,V>::Node::isLeaf() const {
	return children_.size() == 0;
}

template<typename K, typename V>
const typename Tree<K,V>::Key& Tree<K,V>::Node::key() const {
	return key_;
}

template<typename K, typename V>
typename Tree<K,V>::Key& Tree<K,V>::Node::key() {
	return key_;
}

template<typename K, typename V>
const typename Tree<K,V>::Value& Tree<K,V>::Node::value() const {
	return value_;
}

template<typename K, typename V>
typename Tree<K,V>::Value& Tree<K,V>::Node::value() {
	return value_;
}

template<typename K, typename V>
const typename Tree<K,V>::Node& Tree<K,V>::Node::child(u32 index) const {
	require_lt(index, children_.size());
	return children_.at(index);
}

template<typename K, typename V>
typename Tree<K,V>::Node& Tree<K,V>::Node::child(u32 index) {
	require_lt(index, children_.size());
	return children_.at(index);
}

template<typename K, typename V>
bool Tree<K,V>::Node::addChild(const Key& key, const Value& value) {
	for (u32 i = 0; i < children_.size(); i++) {
		if (children_.at(i).key() == key) return false;
	}
	children_.push_back(Node(key, value));
	return true;
}
/*** End Node ***/

template<typename K, typename V>
Tree<K,V>::Tree(const Key& rootKey, const Value& rootValue) :
		root_(rootKey, rootValue)
{}

template<typename K, typename V>
void Tree<K,V>::revertPath(const Path& sourcePath, Path& targetPath) const {
	targetPath.resize(sourcePath.size());
	for (u32 i = 0; i < sourcePath.size(); i++) {
		targetPath.at(targetPath.size() - i - 1) = sourcePath.at(i);
	}
}

template<typename K, typename V>
bool Tree<K,V>::keyExists(const Key& key, const Node& node) const {
	if (node.key() == key) {
		return true;
	}
	else {
		for (u32 i = 0; i < node.nChildren(); i++) {
			if (keyExists(key, node.child(i)))
				return true;
		}
	}
	return false;
}

template<typename K, typename V>
bool Tree<K,V>::pathExists(Path& path, const Node& node) const {
	if ((path.size() == 1) && (node.key() == path.back())) {
		return true;
	}
	else if ((path.size() > 1) && (node.key() == path.back())) {
		path.pop_back();
		for (u32 i = 0; i < node.nChildren(); i++) {
			if (path.back() == node.child(i).key()) {
				return pathExists(path, node.child(i));
			}
		}
	}
	return false;
}

template<typename K, typename V>
bool Tree<K,V>::addNode(Path& path, const Key& key, const Value& value, Node& node) {
	if ((path.size() == 1) && (node.key() == path.back())) {
		return node.addChild(key, value);
	}
	else if ((path.size() > 1) && (node.key() == path.back())) {
		path.pop_back();
		for (u32 i = 0; i < node.nChildren(); i++) {
			if (path.back() == node.child(i).key()) {
				return addNode(path, key, value, node.child(i));
			}
		}
	}
	return false;
}

template<typename K, typename V>
bool Tree<K,V>::addPath(Path& path, const Value& defaultValue, Node& node) {
	if (path.size() == 0) {
		return true;
	}
	else if (path.back() != node.key()) {
		return false; // this case only occurs if path does not start at the root node
	}
	else {
		path.pop_back();
		if (path.size() == 0)
			return true;
		for (u32 i = 0; i < node.nChildren(); i++) {
			if (node.child(i).key() == path.back())
				return addPath(path, defaultValue, node.child(i));
		}
		if(!node.addChild(path.back(), defaultValue))
			return false;
		else
			return addPath(path, defaultValue, node.child(node.nChildren() - 1));
	}
}

template<typename K, typename V>
bool Tree<K,V>::setValue(Path& path, const Value& value, Node& node) {
	if ((path.size() == 1) && (node.key() == path.back())) {
		node.value() = value;
		return true;
	}
	else if ((path.size() > 1) && (node.key() == path.back())) {
		path.pop_back();
		for (u32 i = 0; i < node.nChildren(); i++) {
			if (path.back() == node.child(i).key()) {
				return setValue(path, value, node.child(i));
			}
		}
	}
	return false;
}

template<typename K, typename V>
const typename Tree<K,V>::Value& Tree<K,V>::value(Path& path, const Node& node) const {
	if ((path.size() == 1) && (node.key() == path.back())) {
		return node.value();
	}
	else if ((path.size() > 1) && (node.key() == path.back())) {
		path.pop_back();
		for (u32 i = 0; i < node.nChildren(); i++) {
			if (path.back() == node.child(i).key()) {
				return value(path, node.child(i));
			}
		}
	}
	std::cerr << "Error: Tree::value: path does not exist. Abort." << std::endl;
	exit(1);
}

template<typename K, typename V>
typename Tree<K,V>::Value& Tree<K,V>::value(Path& path, Node& node) {
	return const_cast<Value&>(static_cast<const Tree<K,V> &>(*this).value(path, node));
}

template<typename K, typename V>
const typename Tree<K,V>::Node& Tree<K,V>::node(const Path& path) const {
	require(pathExists(path));
	const Node* n = &root_;
	for (u32 i = 1; i < path.size(); i++) {
		for (u32 j = 0; j < n->nChildren(); j++) {
			if (n->child(j).key() == path.at(i)) {
				n = &(n->child(j));
				break;
			}
		}
	}
	return (*n);
}

template<typename K, typename V>
typename Tree<K,V>::Node& Tree<K,V>::node(const Path& path) {
	return const_cast<Node&>(static_cast<const Tree<K,V> &>(*this).node(path));
}

template<typename K, typename V>
bool Tree<K,V>::keyExists(const Key& key) const {
	return keyExists(key, root_);
}

template<typename K, typename V>
bool Tree<K,V>::pathExists(const Path& path) const {
	Path p;
	revertPath(path, p);
	return pathExists(p, root_);
}

template<typename K, typename V>
bool Tree<K,V>::addNode(const Path& path, const Key& key, const Value& value) {
	Path p;
	revertPath(path, p);
	return addNode(p, key, value, root_);
}

template<typename K, typename V>
bool Tree<K,V>::addPath(const Path& path, const Value& defaultValue) {
	Path p;
	revertPath(path, p);
	return addPath(p, defaultValue, root_);
}

template<typename K, typename V>
bool Tree<K,V>::setValue(const Path& path, const Value& value) {
	Path p;
	revertPath(path, p);
	return setValue(p, value, root_);
}

template<typename K, typename V>
const typename Tree<K,V>::Value& Tree<K,V>::value(const Path& path) const {
	Path p;
	revertPath(path, p);
	return value(p, root_);
}

template<typename K, typename V>
typename Tree<K,V>::Value& Tree<K,V>::value(const Path& path) {
	return const_cast<Value&>(static_cast<const Tree<K,V> &>(*this).value(path));
}

} // namespace

#endif /* CORE_TREE_HH_ */
