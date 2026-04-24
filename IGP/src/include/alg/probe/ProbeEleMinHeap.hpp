//
// Created by 13172 on 2024/6/19.
//

#ifndef VECTORSETSEARCH_PROBEELEMINHEAP_HPP
#define VECTORSETSEARCH_PROBEELEMINHEAP_HPP

#include <vector>

namespace VectorSetSearch {

    // max heap in lower bound, max heap in lower bound
    // min heap is >, so max heap is <

    class ItemProbeEle {
    public:
        float score_;
        uint32_t itemID_;
        uint32_t n_qvec_not_refine_;

        ItemProbeEle() = default;

        ItemProbeEle(const float &score, const uint32_t &itemID, const uint32_t &n_qvec_not_refine) {
            this->score_ = score;
            this->itemID_ = itemID;
            this->n_qvec_not_refine_ = n_qvec_not_refine;
        }
    };

    class ProbeEleMinHeap {

        uint32_t max_size_;
        uint32_t cur_size_;
        std::vector<ItemProbeEle> heap_;

        uint32_t n_item_;
        std::vector<char> is_in_heap_;

        struct {
            bool operator()(const ItemProbeEle &l, const ItemProbeEle &r) const {
                if (l.score_ == r.score_) {
                    return l.n_qvec_not_refine_ < r.n_qvec_not_refine_;
                }
                return l.score_ > r.score_;
            }
        } min_cmp;

    public:
        ProbeEleMinHeap() = default;

        ProbeEleMinHeap(const uint32_t &n_item, const uint32_t &max_size) :
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


        void Update(const float item_lb, const uint32_t itemID, const uint32_t n_qvec_not_refine) {
            if (cur_size_ < max_size_) {
                if (is_in_heap_[itemID]) {
                    for (uint32_t i = 0; i < cur_size_; i++) {
                        if (heap_[i].itemID_ == itemID) {
                            assert(item_lb >= heap_[i].score_);
                            heap_[i].score_ = item_lb;
                            return;
                        }
                    }
                } else {
                    heap_[cur_size_] = ItemProbeEle(item_lb, itemID, n_qvec_not_refine);
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
                        if (heap_[i].itemID_ == itemID) {
                            assert(item_lb >= heap_[i].score_);
                            heap_[i].score_ = item_lb;
                            std::make_heap(heap_.begin(), heap_.end(), min_cmp);
                            break;
                        }
                    }
                } else if (!is_in_heap_[itemID] && item_lb > heap_[0].score_) {
                    is_in_heap_[heap_[0].itemID_] = false;
                    std::pop_heap(heap_.begin(), heap_.end(), min_cmp);
                    heap_[max_size_ - 1] = ItemProbeEle(item_lb, itemID, n_qvec_not_refine);
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

        const ItemProbeEle &MinEle() {
            // should force the input call when cur_size_ == max_size_
            assert(cur_size_ == max_size_);

#ifndef NDEBUG
            uint32_t n_smaller = 0;
            for (uint32_t i = 0; i < max_size_; i++) {
                if (heap_[i].score_ < heap_[0].score_) {
                    n_smaller++;
                }
            }
            assert(n_smaller == 0);
#endif

            return heap_[0];
        }

        const ItemProbeEle &LastTwoMinEle() {
            // should force the input call when cur_size_ == max_size_
            assert(cur_size_ == max_size_);

            const ItemProbeEle &last_two_min_ele = heap_[1].score_ < heap_[2].score_ ? heap_[1] : heap_[2];

#ifndef NDEBUG
            std::vector<float> array(cur_size_);
            for(uint32_t i=0;i<max_size_;i++){
                array[i] = heap_[i].score_;
            }
            std::sort(array.begin(), array.end());
            assert(last_two_min_ele.score_ == array[1]);
#endif

            return last_two_min_ele;
        }

        const ItemProbeEle *Data() {
            return heap_.data();
        }

    };

}
#endif //VECTORSETSEARCH_PROBEELEMINHEAP_HPP
