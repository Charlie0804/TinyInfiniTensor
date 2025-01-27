#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        
        // 存储分配的内存块的起始地址
        // 取巧了，自定义了一篇大内存，应该计算后动态申请的
        size_t retAddr = this->peak;
        for (auto it = free_buffers_block_.begin(); it != free_buffers_block_.end(); it++) {
            if (it->second >= size) { 
                retAddr = it->first;
                //
                size_t blockSize = it->second - size;
                this->free_buffers_block_.erase(it);
                if (blockSize > 0) {
                    free_buffers_block_[retAddr + size] = blockSize;
                }
                this->used += size;
                this->peak = std::max(this->used, this->peak);
                return retAddr;
            }
        }

        return 0;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        size_t retAddr = addr;
        size_t tailAddr = addr + size;
        
    
        // 合并右侧空闲块
        auto mergeRight = [&]() {
            auto it = free_buffers_block_.lower_bound(addr);
            while (it != free_buffers_block_.end() && it->first <= tailAddr) {
                tailAddr = it->first + it->second;
                this->used += it->second;
                it = free_buffers_block_.erase(it);
            }
        };

        // 合并左侧空闲块
        auto mergeLeft = [&]() {
            auto it = free_buffers_block_.lower_bound(addr);
            if (it != free_buffers_block_.begin()) {
                auto prevIt = std::prev(it);
                if (prevIt->first + prevIt->second >= addr) {
                    retAddr = prevIt->first;
                    this->used -= prevIt->second;
                    free_buffers_block_.erase(prevIt);
                }
            }
        };

        mergeRight();
        mergeLeft();

        free_buffers_block_[retAddr] = tailAddr - retAddr;
        this->used -= tailAddr - retAddr;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    // 内存对齐
    size_t Allocator::getAlignedSize(size_t size) {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
