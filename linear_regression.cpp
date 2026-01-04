#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>

// DataPoint struct to represent a single data point
struct DataPoint {
    double x;  // Independent variable (Years of Experience)
    double y;  // Dependent variable (Salary)
    
    DataPoint(double x_val, double y_val) : x(x_val), y(y_val) {}
};

// Abstract base class for machine learning models
class BaseModel {
public:
    // Pure virtual function for training the model
    virtual void train(const std::vector<DataPoint>& data) = 0;
    
    // Pure virtual function for making predictions
    virtual double predict(double x) = 0;
    
    // Virtual destructor for proper cleanup
    virtual ~BaseModel() = default;
};

// Linear Regression class inheriting from BaseModel
class LinearRegression : public BaseModel {
private:
    double slope;      // m in y = mx + b
    double intercept;  // b in y = mx + b
    bool is_trained;   // Flag to check if model is trained

public:
    // Constructor
    LinearRegression() : slope(0.0), intercept(0.0), is_trained(false) {}
    
    // Implementation of train method using Ordinary Least Squares
    void train(const std::vector<DataPoint>& data) override {
        if (data.empty()) {
            std::cerr << "Error: No data provided for training!" << std::endl;
            return;
        }
        
        int n = data.size();
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x_squared = 0;
        
        // Calculate sums for OLS formula
        for (const auto& point : data) {
            sum_x += point.x;
            sum_y += point.y;
            sum_xy += point.x * point.y;
            sum_x_squared += point.x * point.x;
        }
        
        // OLS formulas:
        // slope = (n*Σ(xy) - Σ(x)*Σ(y)) / (n*Σ(x²) - (Σ(x))²)
        // intercept = (Σ(y) - slope*Σ(x)) / n
        
        double denominator = n * sum_x_squared - sum_x * sum_x;
        
        if (std::abs(denominator) < 1e-10) {
            std::cerr << "Error: Cannot compute regression (denominator too small)!" << std::endl;
            return;
        }
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator;
        intercept = (sum_y - slope * sum_x) / n;
        is_trained = true;
        
        std::cout << "Model trained successfully!" << std::endl;
    }
    
    // Implementation of predict method
    double predict(double x) override {
        if (!is_trained) {
            std::cerr << "Error: Model not trained yet!" << std::endl;
            return 0.0;
        }
        return slope * x + intercept;
    }
    
    // Getter methods for slope and intercept
    double getSlope() const { return slope; }
    double getIntercept() const { return intercept; }
    bool isTrained() const { return is_trained; }
};

// CSV Reader class to handle file operations
class CSVReader {
public:
    // Method to read CSV file and return vector of DataPoints
    std::vector<DataPoint> readData(const std::string& filename) {
        std::vector<DataPoint> data;
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return data;
        }
        
        std::string line;
        bool first_line = true;
        
        // Read file line by line
        while (std::getline(file, line)) {
            // Skip header line
            if (first_line) {
                first_line = false;
                continue;
            }
            
            // Parse CSV line
            std::stringstream ss(line);
            std::string x_str, y_str;
            
            if (std::getline(ss, x_str, ',') && std::getline(ss, y_str)) {
                try {
                    double x = std::stod(x_str);
                    double y = std::stod(y_str);
                    data.emplace_back(x, y);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Skipping invalid line: " << line << std::endl;
                }
            }
        }
        
        file.close();
        std::cout << "Successfully loaded " << data.size() << " data points from " << filename << std::endl;
        return data;
    }
};

// Main function demonstrating the complete workflow
int main() {
    std::cout << "=== Linear Regression OOP Project ===" << std::endl;
    std::cout << "Predicting Salary based on Years of Experience" << std::endl << std::endl;
    
    // Step 1: Create CSVReader instance and load data
    CSVReader reader;
    std::vector<DataPoint> dataset = reader.readData("data.csv");
    
    if (dataset.empty()) {
        std::cerr << "No data loaded. Exiting..." << std::endl;
        return 1;
    }
    
    // Step 2: Create LinearRegression object
    LinearRegression model;
    
    // Step 3: Train the model
    std::cout << "\nTraining the model..." << std::endl;
    model.train(dataset);
    
    if (!model.isTrained()) {
        std::cerr << "Model training failed. Exiting..." << std::endl;
        return 1;
    }
    
    // Step 4: Display calculated parameters
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n=== Model Parameters ===" << std::endl;
    std::cout << "Slope (m): " << model.getSlope() << std::endl;
    std::cout << "Intercept (b): " << model.getIntercept() << std::endl;
    std::cout << "Equation: Salary = " << model.getSlope() << " * Experience + " 
              << model.getIntercept() << std::endl;
    
    // Step 5: Interactive prediction
    std::cout << "\n=== Salary Prediction ===" << std::endl;
    double experience;
    
    while (true) {
        std::cout << "\nEnter years of experience (or -1 to exit): ";
        std::cin >> experience;
        
        if (experience == -1) {
            break;
        }
        
        if (experience < 0) {
            std::cout << "Please enter a valid positive number." << std::endl;
            continue;
        }
        
        double predicted_salary = model.predict(experience);
        std::cout << "Predicted salary for " << experience 
                  << " years of experience: $" << predicted_salary << std::endl;
    }
    
    std::cout << "\nThank you for using the Linear Regression predictor!" << std::endl;
    return 0;
}