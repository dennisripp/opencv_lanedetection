#include <vector>
#include <iostream>

enum class Curvature {
    CURVED = 1,
    STRAIGHT = 2,
    UNDETERMINED = 3
};

enum class Direction {
    LEFT = 1,
    RIGHT = 2,
    UNDETERMINED = 3
};

std::string getCurvatureString(Curvature curvature);

std::string getDirectionString(Direction direction);