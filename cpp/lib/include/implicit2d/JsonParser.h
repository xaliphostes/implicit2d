#pragma once
#include <vector>
#include <string>

// Structure to hold geological features
struct GeologicalFeature {
    std::string type;
    std::vector<double> coords;
};

// Function to parse the entire JSON file
std::vector<GeologicalFeature> parseGeologicalJson(const std::string& filename) ;

// int main() {
//     try {
//         auto features = parseGeologicalJson("drawing.json");
        
//         // Print parsed data
//         for (const auto& feature : features) {
//             std::cout << "Type: " << feature.type << std::endl;
//             std::cout << "Coordinates: ";
//             for (size_t i = 0; i < feature.coords.size(); i += 2) {
//                 std::cout << "(" << feature.coords[i] << ", " << feature.coords[i + 1] << ") ";
//             }
//             std::cout << "\n\n";
//         }
        
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//     }
    
//     return 0;
// }