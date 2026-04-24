//
// Created by 13172 on 2024/2/29.
//

#ifndef VECTORSETSEARCH_POSITIONHEAP_HPP
#define VECTORSETSEARCH_POSITIONHEAP_HPP

#include <vector>

namespace VectorSetSearch {

    // max heap in lower bound, max heap in lower bound
    // min heap is >, so max heap is <

    class PositionMinHeap {

        uint32_t max_size_;
        uint32_t cur_size_;
        std::vector<std::pair<float, uint32_t>> heap_;

        uint32_t n_item_;
        std::vector<char> is_in_heap_;


        struct {
            bool operator()(const std::pair<float, uint32_t> &l, const std::pair<float, uint32_t> &r) const {
                return l.first > r.first;
            }
        } min_cmp;

    public:
        PositionMinHeap() = default;

        PositionMinHeap(const uint32_t &n_item, const uint32_t &max_size) :
                n_item_(n_item), cur_size_(0), max_size_(max_size) {
            heap_.resize(max_size_);
            is_in_heap_.resize(n_item_);
            is_in_heap_.assign(n_item_, false);
        }

        uint32_t Size() {
            assert(cur_size_ <= heap_.size());
            return cur_size_;
        }

        void Reset() {
            cur_size_ = 0;
            is_in_heap_.assign(n_item_, false);
        }


        void Update(const float &item_lb, const uint32_t &itemID) {
            if (cur_size_ < max_size_) {
                if (is_in_heap_[itemID]) {
                    for (uint32_t i = 0; i < cur_size_; i++) {
                        if (heap_[i].second == itemID) {
                            assert(item_lb >= heap_[i].first);
                            heap_[i].first = item_lb;
                            return;
                        }
                    }
                } else {
                    heap_[cur_size_] = std::make_pair(item_lb, itemID);
                    is_in_heap_[itemID] = true;
                    cur_size_++;
                    if (cur_size_ == max_size_) {
                        std::make_heap(heap_.begin(), heap_.end(), min_cmp);
                    }
                }
            } else {
                assert(cur_size_ == max_size_);
                if (is_in_heap_[itemID]) {
                    for (uint32_t i = 0; i < max_size_; i++) {
                        if (heap_[i].second == itemID) {
                            assert(item_lb >= heap_[i].first);
                            heap_[i].first = item_lb;
                            std::make_heap(heap_.begin(), heap_.end(), min_cmp);
                            break;
                        }
                    }
                } else if (!is_in_heap_[itemID] && item_lb > heap_[0].first) {
                    is_in_heap_[heap_[0].second] = false;
                    std::pop_heap(heap_.begin(), heap_.end(), min_cmp);
                    heap_[max_size_ - 1] = std::make_pair(item_lb, itemID);
                    is_in_heap_[itemID] = true;
                    std::push_heap(heap_.begin(), heap_.end(), min_cmp);
                }
#ifndef NDEBUG
                uint32_t count = 0;
                for (uint32_t tmp_itemID = 0; tmp_itemID < n_item_; tmp_itemID++) {
                    if (is_in_heap_[tmp_itemID]) {
                        count++;
                    }
                }
                assert(count == cur_size_);
#endif
            }

        }

        std::pair<float, uint32_t> Top() {
            if (cur_size_ < max_size_) {
                spdlog::error("The heap is not full, the top item is not valid");
                exit(-1);
            } else {
                assert(cur_size_ == max_size_);
                return heap_[0];
            }
        }

        const std::pair<float, uint32_t> *Data() {
            return heap_.data();
        }

    };

}
#endif //VECTORSETSEARCH_POSITIONHEAP_HPP
