#include "PointCloudChangeDetector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <tuple>
#include <unordered_map>
#include <pcl/common/point_tests.h>

namespace
{
using Point = pcl::PointXYZRGB;
using Cloud = PointCloudChangeDetector::Cloud;

// Power-of-two grids and 50/50 holdout scoring from ChangeDetect's
// changeDetectAdaptiveByHoldout. Voxel counts persist as keyframes are fused.
constexpr double maxFalseAddRate = 0.20;
constexpr int maxGlobalDepth = 12;
constexpr int defaultLevel = 3;
constexpr int levelCount = 5;
constexpr int blockOffset = 3;

struct VoxelKey
{
    double x;
    double y;
    double z;

    bool operator==(const VoxelKey& other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }

    VoxelKey Coarsen(double ratio) const
    {
        return {std::floor(x * ratio), std::floor(y * ratio), std::floor(z * ratio)};
    }
};

struct VoxelKeyHash
{
    std::size_t operator()(const VoxelKey& key) const
    {
        const std::hash<double> hash;
        std::size_t seed = hash(key.x);
        seed ^= hash(key.y) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        seed ^= hash(key.z) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        return seed;
    }
};

VoxelKey KeyFor(const Point& point, double inverseEdge)
{
    return {std::floor(point.x * inverseEdge),
            std::floor(point.y * inverseEdge),
            std::floor(point.z * inverseEdge)};
}

struct HoldoutScore
{
    std::size_t pointCount = 0;
    long double expectedFalseAdditions = 0.0;

    static long double Expected(std::size_t count)
    {
        // Every other point in the voxel must also land in the held-out half.
        return count <= 1075
            ? count * std::ldexp(1.0, 1 - static_cast<int>(count))
            : 0.0;
    }

    void ReplaceCount(std::size_t oldCount, std::size_t newCount)
    {
        pointCount = pointCount - oldCount + newCount;
        expectedFalseAdditions += Expected(newCount) - Expected(oldCount);
    }

    bool MeetsTarget() const
    {
        return expectedFalseAdditions <= maxFalseAddRate * pointCount;
    }
};

class VoxelLayer
{
public:
    VoxelLayer(int exponent, const Cloud& cloud)
        : mExponent(exponent), mInverseEdge(std::ldexp(1.0, -exponent))
    {
        for (const auto& point : cloud)
            ++mCounts[KeyFor(point, mInverseEdge)];
        for (const auto& voxel : mCounts)
            mGlobalScore.ReplaceCount(0, voxel.second);
    }

    bool MeetsGlobalTarget() const { return mGlobalScore.MeetsTarget(); }

    void PrepareBlocks(int blockExponent)
    {
        const double ratio = std::ldexp(1.0, mExponent - blockExponent);
        if (mBlockRatio == ratio)
            return;

        mBlockRatio = ratio;
        mBlocks.clear();
        for (const auto& voxel : mCounts)
            mBlocks[voxel.first.Coarsen(ratio)].ReplaceCount(0, voxel.second);
    }

    bool MeetsBlockTarget(const VoxelKey& block) const
    {
        const auto score = mBlocks.find(block);
        return score != mBlocks.end() && score->second.MeetsTarget();
    }

    bool Occupied(const Point& point) const
    {
        return mCounts.count(KeyFor(point, mInverseEdge)) != 0;
    }

    void Update(const Point& point, bool adding)
    {
        const VoxelKey voxel = KeyFor(point, mInverseEdge);
        auto& count = adding ? mCounts[voxel] : mCounts.find(voxel)->second;
        const std::size_t oldCount = count;
        const std::size_t newCount = adding ? oldCount + 1 : oldCount - 1;
        count = newCount;
        if (newCount == 0)
            mCounts.erase(voxel);

        mGlobalScore.ReplaceCount(oldCount, newCount);
        const VoxelKey block = voxel.Coarsen(mBlockRatio);
        auto& score = adding ? mBlocks[block] : mBlocks.find(block)->second;
        score.ReplaceCount(oldCount, newCount);
        if (score.pointCount == 0)
            mBlocks.erase(block);
    }

private:
    int mExponent;
    double mInverseEdge;
    double mBlockRatio = 0.0;
    HoldoutScore mGlobalScore;
    std::unordered_map<VoxelKey, std::size_t, VoxelKeyHash> mCounts;
    std::unordered_map<VoxelKey, HoldoutScore, VoxelKeyHash> mBlocks;
};

struct Bounds
{
    std::array<double, 3> minimum;
    std::array<double, 3> maximum;

    Bounds()
    {
        minimum.fill(std::numeric_limits<double>::infinity());
        maximum.fill(-std::numeric_limits<double>::infinity());
    }

    void Extend(const Point& point)
    {
        const std::array<double, 3> coordinates{{point.x, point.y, point.z}};
        for (std::size_t axis = 0; axis < coordinates.size(); ++axis)
        {
            minimum[axis] = std::min(minimum[axis], coordinates[axis]);
            maximum[axis] = std::max(maximum[axis], coordinates[axis]);
        }
    }

    double Span() const
    {
        return std::max({maximum[0] - minimum[0], maximum[1] - minimum[1],
                         maximum[2] - minimum[2]});
    }
};
}

struct PointCloudChangeDetector::State
{
    Cloud cloud;
    Bounds bounds;
    std::map<int, VoxelLayer> layers;
    int globalExponent = 0;

    VoxelLayer& Layer(int exponent)
    {
        const auto layer = layers.lower_bound(exponent);
        if (layer != layers.end() && layer->first == exponent)
            return layer->second;
        return layers.emplace_hint(layer, std::piecewise_construct,
                                   std::forward_as_tuple(exponent),
                                   std::forward_as_tuple(exponent, cloud))->second;
    }

    int SelectGlobalExponent()
    {
        const double span = bounds.Span();
        if (span <= 0.0)
            return -5; // ChangeDetect's 0.03125 fallback for a zero span.

        const int coarsest = static_cast<int>(std::ceil(std::log2(span)));
        const int finest = coarsest - maxGlobalDepth;
        int exponent = std::max(finest, std::min(globalExponent, coarsest));

        // The false-add score decreases as cells merge. Start at the previous
        // scale and look for the finest passing level in the same search range.
        while (exponent < coarsest && !Layer(exponent).MeetsGlobalTarget())
            ++exponent;
        while (exponent > finest && Layer(exponent - 1).MeetsGlobalTarget())
            --exponent;
        return exponent;
    }
};

PointCloudChangeDetector::PointCloudChangeDetector() : mState(new State) {}
PointCloudChangeDetector::~PointCloudChangeDetector() = default;

const PointCloudChangeDetector::Cloud& PointCloudChangeDetector::GetCloud() const
{
    return mState->cloud;
}

void PointCloudChangeDetector::AppendChanges(const Cloud& current)
{
    State& state = *mState;
    const std::size_t previousSize = state.cloud.size();
    if (previousSize == 0)
    {
        for (const auto& point : current)
            if (pcl::isFinite(point))
                state.cloud.push_back(point);
    }
    else if (!current.empty())
    {
        state.globalExponent = state.SelectGlobalExponent();
        const int first = state.globalExponent - defaultLevel;
        const int blockExponent = state.globalExponent + blockOffset;
        const double inverseBlockEdge = std::ldexp(1.0, -blockExponent);

        std::array<VoxelLayer*, levelCount> levels;
        for (int level = 0; level < levelCount; ++level)
        {
            levels[level] = &state.Layer(first + level);
            levels[level]->PrepareBlocks(blockExponent);
        }
        state.layers.erase(state.layers.begin(), state.layers.lower_bound(first));
        state.layers.erase(state.layers.lower_bound(first + levelCount), state.layers.end());

        // All points in a frame compare against the same reference occupancy.
        // Cache each block's finest passing level for the duration of the frame.
        std::unordered_map<VoxelKey, int, VoxelKeyHash> blockLevels;
        for (const auto& point : current)
        {
            if (!pcl::isFinite(point))
                continue;
            const VoxelKey block = KeyFor(point, inverseBlockEdge);
            auto level = blockLevels.find(block);
            if (level == blockLevels.end())
            {
                int selectedLevel = defaultLevel;
                for (int candidate = 0; candidate < levelCount; ++candidate)
                {
                    if (levels[candidate]->MeetsBlockTarget(block))
                    {
                        selectedLevel = candidate;
                        break;
                    }
                }
                level = blockLevels.emplace(block, selectedLevel).first;
            }
            if (!levels[level->second]->Occupied(point))
                state.cloud.push_back(point);
        }
    }

    for (std::size_t i = previousSize; i < state.cloud.size(); ++i)
    {
        const auto& point = state.cloud[i];
        state.bounds.Extend(point);
        for (auto& layer : state.layers)
            layer.second.Update(point, true);
    }
}

void PointCloudChangeDetector::Truncate(std::size_t pointCount)
{
    State& state = *mState;
    if (pointCount >= state.cloud.size())
        return;

    if (pointCount == 0)
    {
        state.layers.clear();
        state.globalExponent = 0;
    }
    else
    {
        for (std::size_t i = pointCount; i < state.cloud.size(); ++i)
            for (auto& layer : state.layers)
                layer.second.Update(state.cloud[i], false);
    }
    state.cloud.resize(pointCount);
    state.bounds = Bounds{};
    for (const auto& point : state.cloud)
        state.bounds.Extend(point);
}
