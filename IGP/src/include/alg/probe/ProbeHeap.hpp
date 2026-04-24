//
// Created by 13172 on 2024/4/15.
//

#ifndef VECTORSETSEARCH_PROBEHEAP_HPP
#define VECTORSETSEARCH_PROBEHEAP_HPP

#include <cstdint>
#include <vector>

namespace VectorSetSearch {

    class Probe {
    public:
        uint32_t qvecID, nprobe;
        float score;

        Probe() = default;

        Probe(const uint32_t &qvecID, const uint32_t &nprobe, const float &score) :
                qvecID(qvecID), nprobe(nprobe), score(score) {}
    };

    // max heap in lower bound, max heap in lower bound
    // min heap is >, so max heap is <
    class ProbeMaxHeap {

        uint32_t max_size_;
        uint32_t cur_size_;
        std::vector<Probe> heap_;

        struct {
            bool operator()(const Probe &l, const Probe &r) const {
                return l.score < r.score;
            }
        } cmp;

    public:
        ProbeMaxHeap() = default;

        ProbeMaxHeap(const uint32_t &max_size) : cur_size_(0), max_size_(max_size) {
            heap_.resize(max_size_);
        }

        uint32_t Size() {
            assert(cur_size_ <= heap_.size());
            return cur_size_;
        }

        void Reset() {
            cur_size_ = 0;
        }

        void Update(const uint32_t &qvecID, const uint32_t &nprobe, const float &score) {
            if (cur_size_ < max_size_) {
                heap_[cur_size_] = Probe(qvecID, nprobe, score);
                cur_size_++;
                if (cur_size_ == max_size_) {
                    std::make_heap(heap_.begin(), heap_.end(), cmp);
                }
            } else {
                assert(cur_size_ == max_size_);
                assert(heap_[0].qvecID == qvecID && heap_[0].nprobe + 1 == nprobe && heap_[0].score >= score);
                std::pop_heap(heap_.begin(), heap_.end(), cmp);
                heap_[max_size_ - 1] = Probe(qvecID, nprobe, score);
                std::push_heap(heap_.begin(), heap_.end(), cmp);
            }

        }

        void Pop() {
            std::pop_heap(heap_.begin(), heap_.end(), cmp);
            cur_size_--;
        }

        Probe Top() {
            assert(cur_size_ <= max_size_);
            return heap_[0];
        }

        const Probe *Data() {
            return heap_.data();
        }

    };

}
#endif //VECTORSETSEARCH_PROBEHEAP_HPP
