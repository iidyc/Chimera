#pragma once

#include <queue>

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)

#endif

typedef size_t labeltype;

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif
namespace hnswlib_probe {
    //typedef void *labeltype;
    //typedef float(*DISTFUNC) (const void *, const void *, const void *);
    template<typename MTYPE>
    using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);


    template<typename dist_t>
    class AlgorithmInterface {
    public:
        //virtual void addPoint(void *, labeltype) = 0;
//        virtual std::priority_queue<std::pair<dist_t, labeltype >> searchKnn(void *, int) = 0;
    };

    template<typename MTYPE>
    class SpaceInterface {
    public:
        //virtual void search(void *);
        virtual size_t get_data_size() = 0;

        virtual DISTFUNC<MTYPE> get_dist_func() = 0;

        virtual void *get_dist_func_param() = 0;

    };

    class SearchResultEle {
    public:
        float score_;
        uint32_t itemID_;

        SearchResultEle() = default;

        SearchResultEle(const float &score, const uint32_t &itemID) {
            this->score_ = score;
            this->itemID_ = itemID;
        }
    };

    class SearchResultMinHeap {
    public:
        // n_centroid is the maximum size of the vector
        uint32_t search_max_size_, n_centroid_;
        uint32_t cur_size_;
        std::vector<SearchResultEle> heap_;
        std::vector<char> is_in_heap; // n_centroid

        struct {
            bool operator()(const SearchResultEle &l, const SearchResultEle &r) const {
                if (l.score_ == r.score_) {
                    return l.itemID_ > r.itemID_;
                }
                return l.score_ > r.score_;
            }
        } min_cmp;


        SearchResultMinHeap() = default;

        SearchResultMinHeap(const uint32_t search_max_size, const uint32_t n_centroid) :
                cur_size_(0), search_max_size_(search_max_size), n_centroid_(n_centroid) {
            heap_.resize(search_max_size);
            is_in_heap.resize(n_centroid, false);
        }

        uint32_t Size() {
            assert(cur_size_ <= search_max_size_);
            return cur_size_;
        }

        void Reset(const uint32_t search_size) {
            cur_size_ = 0;
            search_max_size_ = search_size;
            is_in_heap.assign(n_centroid_, false);
        }

        inline void Update(const float score, const uint32_t centID) {
            assert(is_in_heap[centID] == false);
            if (cur_size_ < search_max_size_) {
                heap_[cur_size_] = SearchResultEle(score, centID);
                cur_size_++;
                std::push_heap(heap_.begin(), heap_.begin() + cur_size_, min_cmp);
                //                if (cur_size_ == search_max_size_) {
                //                    std::make_heap(heap_.begin(), heap_.end(), min_cmp);
                //                }
                is_in_heap[centID] = true;
            } else {
                assert(cur_size_ == search_max_size_);
                if (score > heap_[0].score_) {
                    is_in_heap[heap_[0].itemID_] = false;
                    std::pop_heap(heap_.begin(), heap_.begin() + search_max_size_, min_cmp);
                    heap_[search_max_size_ - 1] = SearchResultEle(score, centID);
                    std::push_heap(heap_.begin(), heap_.begin() + search_max_size_, min_cmp);
                    is_in_heap[centID] = true;
                }
            }
        }

        bool InHeap(const uint32_t centID) {
            return is_in_heap[centID];
        }

        SearchResultEle Top() {
            // should force the input call when cur_size_ == max_size_
            //            if (cur_size_ < search_max_size_) {
            //                return SearchResultEle(-std::numeric_limits<float>::max(), n_centroid_ + 1);
            //            }
            assert(cur_size_ == search_max_size_);

#ifndef NDEBUG
            uint32_t n_smaller = 0;
            for (uint32_t i = 1; i < cur_size_; i++) {
                if (heap_[i].score_ < heap_[0].score_ - 1e-3) {
                    n_smaller++;
                }
            }
            assert(n_smaller == 0);
#endif

            return heap_[0];
        }

        std::vector<SearchResultEle> PopTopScore(const uint32_t topk) {
            const uint32_t heap_max_size = std::min(cur_size_, search_max_size_);

#ifndef NDEBUG
            for (uint32_t candID = 0; candID < heap_max_size; candID++) {
                const uint32_t itemID = heap_[candID].itemID_;
                assert(is_in_heap[itemID]);
            }

            std::set<uint32_t> itemID_s;
            for (uint32_t candID = 0; candID < heap_max_size; candID++) {
                const uint32_t itemID = heap_[candID].itemID_;
                itemID_s.insert(itemID);
            }
            assert(itemID_s.size() == heap_max_size);
#endif

            std::sort(heap_.begin(), heap_.begin() + heap_max_size,
                      [](const SearchResultEle l, const SearchResultEle r) {
                          if (l.score_ == r.score_) {
                              return l.itemID_ > r.itemID_;
                          } else {
                              return l.score_ > r.score_;
                          }
                      });

            const uint32_t n_pop = std::min(topk, heap_max_size);
            std::vector<SearchResultEle> result(n_pop);
            for (uint32_t candID = 0; candID < n_pop; candID++) {
                result[candID] = heap_[candID];
                assert(result[candID].itemID_ == heap_[candID].itemID_);
                const uint32_t itemID = result[candID].itemID_;
                assert(is_in_heap[itemID]);
                is_in_heap[itemID] = false;
            }

            const uint32_t n_remain = heap_max_size - n_pop;
            for (uint32_t candID = 0; candID < n_remain; candID++) {
                const uint32_t itemID = heap_[candID + n_pop].itemID_;
                assert(is_in_heap[itemID]);
                heap_[candID] = heap_[candID + n_pop];
            }
            cur_size_ = n_remain;
            std::make_heap(heap_.begin(), heap_.begin() + cur_size_, min_cmp);

            return result;
        }

        const SearchResultEle *Data() {
            return heap_.data();
        }
    };

}

#include "L2space.hpp"
#include "hnswalg.hpp"
