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

/*
 * HashMap.hh
 *
 *  Created on: Jun 12, 2017
 *      Author: richard
 */

#ifndef CORE_HASHMAP_HH_
#define CORE_HASHMAP_HH_

#include <Core/CommonHeaders.hh>
#include <algorithm>
#include <list>

namespace Core {

/*
 * HashMap
 * @typename K key class, needs to have an == operator of the form bool K::operator==(const K& key) const { ... }
 * @typename V value class
 */
template<typename K, typename V>
class HashMap
{
public:
	class Entry {
	public:
		K key;
		V value;
		Entry* next;
		Entry* prev;
	public:
		Entry(const K& key, const V& value) : key(key), value(value), next(0), prev(0) {}
	};
private:
	std::vector< Entry* > bucketStartNode_;
	std::vector< Entry* > bucketEndNode_;
	Entry* head_;
	u32 (*hashFunction_)(const K&);
	u32 size_;
	u32 hash(const K& key) const;
public:
	/*
	 * @param hashFunction function of the form hashFunction(const K&) to generate an integer from the Key
	 * @param nBuckets hash table size, a prime is recommended
	 */
	HashMap(u32 (*hashFunction)(const K&), u32 nBuckets = 15331);
	virtual ~HashMap();
	/*
	 * clear the hash map
	 * @param tableSize new size of the hash table (number of buckets)
	 */
	void clear();
	/*
	 * @return number of elements in the hash map
	 */
	u32 size() const { return size_; }
	/*
	 * @param key the key to look up
	 * @return pointer to the object associated with the given key or 0 if key not found
	 */
	Entry* find(const K& key);
	/*
	 * @param key the key to use for the value. If key already exists, it will be overwritten
	 * @param value the value to be inserted into the list
	 * @return a pointer to the inserted entry
	 */
	Entry* insert(const K& key, const V& value);
	/*
	 * @param key the key of the entry to be removed from the hash map
	 */
	void remove(const K& key);
	/*
	 * @return pointer to first entry in the list
	 */
	Entry* begin() { return head_; }
	/*
	 * @return a null pointer
	 */
	Entry* end() { return 0; }
};

template<typename K, typename V>
HashMap<K,V>::HashMap(u32(*hashFunction)(const K&), u32 nBuckets) :
	bucketStartNode_(nBuckets, 0),
	bucketEndNode_(nBuckets, 0),
	head_(0),
	hashFunction_(hashFunction),
	size_(0)
{}

template<typename K, typename V>
HashMap<K,V>::~HashMap() {
	clear();
}

template<typename K, typename V>
u32 HashMap<K,V>::hash(const K& key) const {
	u32 h = hashFunction_(key);
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = (h >> 16) ^ h;
    return h % bucketStartNode_.size();
}

template<typename K, typename V>
void HashMap<K,V>::clear() {
	while (head_ != 0) {
		Entry* entry = head_;
		head_ = head_->next;
		delete entry;
	}
	for (u32 i = 0; i < bucketStartNode_.size(); i++) {
		bucketStartNode_[i] = 0;
		bucketEndNode_[i] = 0;
	}
	size_ = 0;
}

template<typename K, typename V>
typename HashMap<K,V>::Entry* HashMap<K,V>::find(const K& key) {
	u32 h = hash(key);
	if (bucketStartNode_[h] == 0) return 0;
	for (Entry* entry = bucketStartNode_[h]; entry != bucketEndNode_[h]->next; entry = entry->next) {
		if (entry->key == key)
			return entry;
	}
	return 0;
}

template<typename K, typename V>
typename HashMap<K,V>::Entry* HashMap<K,V>::insert(const K& key, const V& value) {
	Entry* entry = find(key);
	if (entry == 0) {
		// insert new entry
		u32 h = hash(key);
		entry = new Entry(key, value);
		if (bucketStartNode_[h] == 0) {
			entry->next = head_;
			if (head_ != 0) head_->prev = entry;
			bucketEndNode_[h] = entry;
			head_ = entry;
		}
		else {
			entry->next = bucketStartNode_[h];
			entry->prev = bucketStartNode_[h]->prev;
			bucketStartNode_[h]->prev = entry;
			if (entry->prev != 0) entry->prev->next = entry;
			if (bucketStartNode_[h] == head_)
				head_ = entry;
		}
		bucketStartNode_[h] = entry;
		size_++;
	}
	else {
		entry->value = value;
	}
	return entry;
}

template<typename K, typename V>
void HashMap<K,V>::remove(const K& key) {
	Entry* entry = find(key);
	u32 h = hash(key);
	if (entry != 0) {
		if (entry->next != 0) entry->next->prev = entry->prev;
		if (entry->prev != 0) entry->prev->next = entry->next;
		if (entry == head_) head_ = entry->next;
		if (bucketStartNode_[h] == bucketEndNode_[h]) bucketStartNode_[h] = bucketEndNode_[h] = 0;
		else if (entry == bucketStartNode_[h]) bucketStartNode_[h] = entry->next;
		else if (entry == bucketEndNode_[h]) bucketEndNode_[h] = entry->prev;
		delete entry;
		size_--;
	}
}

} // namespace

#endif /* CORE_HASHMAP_HH_ */
