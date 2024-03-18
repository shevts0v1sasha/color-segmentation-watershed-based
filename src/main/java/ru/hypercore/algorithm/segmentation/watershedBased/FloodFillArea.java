package ru.hypercore.algorithm.segmentation.watershedBased;

import org.opencv.core.Point;

import java.util.Arrays;
import java.util.Set;

public record FloodFillArea(Long id, Point seedPoint, double[] color, Set<FloodFillArea> neighbors, FloodFillResult floodFillResult) {
    public void addNeighbor(FloodFillArea neighbor) {
        this.neighbors.add(neighbor);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        FloodFillArea that = (FloodFillArea) o;

        return id.equals(that.id);
    }

    @Override
    public int hashCode() {
        return id.hashCode();
    }

    @Override
    public String toString() {
        return "FloodFillArea{" +
                "id=" + id +
                ", seedPoint=" + seedPoint +
                ", color=" + Arrays.toString(color) +
                ", neighbors size=" + neighbors.size() +
                '}';
    }
}