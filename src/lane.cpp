#include "../header/lane.hpp"

// std::string LaneName(Lane lane) {
//     switch (lane) {
//         case Lane::STRAIGHTLEFT: return "STRAIGHTLEFT";
//         case Lane::STRAIGHTRIGHT: return "STRAIGHTRIGHT";
//         case Lane::CURVED: return "CURVED";
//         case Lane::STRAIGHT: return "STRAIGHT";
//     }
//     return "UNKNOWN";
// }

std::string getCurvatureString(Curvature curvature) {
    switch (curvature) {
        case Curvature::CURVED:
            return "Curved";
        case Curvature::STRAIGHT:
            return "Straight";
        case Curvature::UNDETERMINED:
            return "Undetermined";
        default:
            return "Unknown";
    }
}

std::string getDirectionString(Direction direction) {
    switch (direction) {
        case Direction::LEFT:
            return "Left";
        case Direction::RIGHT:
            return "Right";
        case Direction::UNDETERMINED:
            return "Undetermined";
        default:
            return "Unknown";
    }
}