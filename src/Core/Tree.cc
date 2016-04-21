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

using namespace Core;

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
	return key_;
}

template<typename K, typename V>
typename Tree<K,V>::Value& Tree<K,V>::Node::value() {
	return key_;
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
	else if (path.size() > 1) {
		for (u32 i = 0; i < node.nChildren(); i++) {
			if (path.back() == node.child(i).key()) {
				path.pop_back();
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
	else if (path.size() > 1) {
		for (u32 i = 0; i < node.nChildren(); i++) {
			if (path.back() == node.child(i).key()) {
				path.pop_back();
				return addNode(path, key, value, node.child(i));
			}
		}
	}
	return false;
}

template<typename K, typename V>
bool Tree<K,V>::setValue(Path& path, const Value& value, Node& node) {
	if ((path.size() == 1) && (node.key() == path.back())) {
		node.value() = value;
		return true;
	}
	else if (path.size() > 1) {
		for (u32 i = 0; i < node.nChildren(); i++) {
			if (path.back() == node.child(i).key()) {
				path.pop_back();
				return setValue(path, value, node.child(i));
			}
		}
	}
	return false;
}

template<typename K, typename V>
const typename Tree<K,V>::Value& Tree<K,V>::getValue(Path& path, const Node& node) const {
	if ((path.size() == 1) && (node.key() == path.back())) {
		return node.value();
	}
	else if (path.size() > 1) {
		for (u32 i = 0; i < node.nChildren(); i++) {
			if (path.back() == node.child(i).key()) {
				path.pop_back();
				return getValue(path, node.child(i));
			}
		}
	}
	std::cerr << "Error: Tree::getValue: path does not exist. Abort." << std::endl;
	exit(1);
}

template<typename K, typename V>
typename Tree<K,V>::Value& Tree<K,V>::getValue(Path& path, Node& node) {
	if ((path.size() == 1) && (node.key() == path.back())) {
		return node.value();
	}
	else if (path.size() > 1) {
		for (u32 i = 0; i < node.nChildren(); i++) {
			if (path.back() == node.child(i).key()) {
				path.pop_back();
				return getValue(path, node.child(i));
			}
		}
	}
	std::cerr << "Error: Tree::getValue: path does not exist. Abort." << std::endl;
	exit(1);
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
bool Tree<K,V>::setValue(const Path& path, const Value& value) {
	Path p;
	revertPath(path, p);
	return setValue(p, value, root_);
}

template<typename K, typename V>
const typename Tree<K,V>::Value& Tree<K,V>::getValue(const Path& path) const {
	Path p;
	revertPath(path, p);
	return getValue(p, root_);
}

template<typename K, typename V>
typename Tree<K,V>::Value& Tree<K,V>::getValue(const Path& path) {
	Path p;
	revertPath(path, p);
	return getValue(p, root_);
}
