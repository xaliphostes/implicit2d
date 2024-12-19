#include <implicit2d/JsonParser.h>
#include <iostream>
#include <fstream>
#include <sstream>

// Function to remove whitespace and newlines
std::string cleanJsonString(const std::string& input) {
    std::string output;
    for (char c : input) {
        if (!std::isspace(c)) output += c;
    }
    return output;
}

// Function to parse the coordinates array
std::vector<double> parseCoordinates(const std::string& jsonStr, size_t& pos) {
    std::vector<double> coords;
    pos = jsonStr.find('[', pos) + 1;
    
    while (pos < jsonStr.length()) {
        // Find the next number
        while (pos < jsonStr.length() && !std::isdigit(jsonStr[pos]) && jsonStr[pos] != '-') {
            if (jsonStr[pos] == ']') return coords;
            pos++;
        }
        
        // Parse the number
        size_t endPos;
        double value = std::stod(jsonStr.substr(pos), &endPos);
        coords.push_back(value);
        pos += endPos;
    }
    return coords;
}

// Function to parse the entire JSON file
std::vector<GeologicalFeature> parseGeologicalJson(const std::string& filename) {
    // Read file
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string jsonStr = cleanJsonString(buffer.str());
    
    std::vector<GeologicalFeature> features;
    size_t pos = 0;
    
    while ((pos = jsonStr.find("\"type\"", pos)) != std::string::npos) {
        GeologicalFeature feature;
        
        // Parse type
        pos = jsonStr.find(':', pos) + 1;
        pos = jsonStr.find('"', pos) + 1;
        size_t endPos = jsonStr.find('"', pos);
        feature.type = jsonStr.substr(pos, endPos - pos);
        
        // Parse coordinates
        pos = jsonStr.find("\"coords\"", pos);
        feature.coords = parseCoordinates(jsonStr, pos);
        
        features.push_back(feature);
    }
    
    return features;
}
