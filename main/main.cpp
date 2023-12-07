// Include necessary TensorFlow Lite headers for model handling
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/string_util.h>

// Include OpenCV headers for image processing
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Other standard C++ libraries
#include <iostream>
#include <fstream>
#include <memory>

// Constants for the keypoints and confidence threshold
const int num_keypoints = 17;
const float confidence_threshold = 0.2;

// Define a vector of pairs to represent connections between keypoints
const std::vector<std::pair<int, int>> connections = {
    {0, 1},     // Connection between keypoint 0 (e.g., nose) and keypoint 1 (e.g., left eye)
    {0, 2},     // Connection between keypoint 0 and keypoint 2 (e.g., right eye)
    {1, 3},     // Connection between keypoint 1 and keypoint 3 (e.g., left shoulder)
    {2, 4},     // Connection between keypoint 2 and keypoint 4 (e.g., right shoulder)
    {5, 6},     // Connection between keypoint 5 (e.g., left hip) and keypoint 6 (e.g., right hip)
    {5, 7},     // Connection between keypoint 5 and keypoint 7 (e.g., left knee)
    {7, 9},     // Connection between keypoint 7 and keypoint 9 (e.g., left ankle)
    {6, 8},     // Connection between keypoint 6 and keypoint 8 (e.g., right knee)
    {8, 10},    // Connection between keypoint 8 and keypoint 10 (e.g., right ankle)
    {5, 11},    // Connection between keypoint 5 and keypoint 11 (e.g., left elbow)
    {6, 12},    // Connection between keypoint 6 and keypoint 12 (e.g., right elbow)
    {11, 12},   // Connection between keypoint 11 and keypoint 12 (e.g., wrists)
    {11, 13},   // Connection between keypoint 11 and keypoint 13 (e.g., left hand)
    {13, 15},   // Connection between keypoint 13 and keypoint 15 (e.g., left fingertips)
    {12, 14},   // Connection between keypoint 12 and keypoint 14 (e.g., right hand)
    {14, 16}    // Connection between keypoint 14 and keypoint 16 (e.g., right fingertips)
};

void draw_keypoints(cv::Mat &resized_image, float *output) {
    // Get the size of the square image (assuming square due to the usage of rows)
    int square_dim = resized_image.rows;

    // Loop through the keypoints detected
    for (int i = 0; i < num_keypoints; ++i) {
        // Extract x, y coordinates and confidence score for each keypoint
        float y = output[i * 3];        // y-coordinate
        float x = output[i * 3 + 1];    // x-coordinate
        float conf = output[i * 3 + 2]; // confidence score

        // Check if the confidence score is above the defined threshold
        if (conf > confidence_threshold) {
            // Calculate the pixel coordinates in the resized image
            int img_x = static_cast<int>(x * square_dim); // Scale x-coordinate
            int img_y = static_cast<int>(y * square_dim); // Scale y-coordinate

            // Draw a circle at the identified keypoint location on the image
            cv::circle(resized_image, cv::Point(img_x, img_y), 2, cv::Scalar(255, 200, 200), 1);
            // Arguments: image, center point, radius, color, thickness
        }
    }
// Draw skeleton connections between keypoints
for (const auto &connection : connections) {
    // Extract indices of keypoints forming the connection
    int index1 = connection.first;
    int index2 = connection.second;

    // Get information for the first keypoint
    float y1 = output[index1 * 3];       // y-coordinate of keypoint 1
    float x1 = output[index1 * 3 + 1];   // x-coordinate of keypoint 1
    float conf1 = output[index1 * 3 + 2];// confidence score of keypoint 1

    // Get information for the second keypoint
    float y2 = output[index2 * 3];       // y-coordinate of keypoint 2
    float x2 = output[index2 * 3 + 1];   // x-coordinate of keypoint 2
    float conf2 = output[index2 * 3 + 2];// confidence score of keypoint 2

    // Check if both keypoints have confidence scores above the threshold
    if (conf1 > confidence_threshold && conf2 > confidence_threshold) {
        // Scale coordinates to fit the resized image
        int img_x1 = static_cast<int>(x1 * square_dim); // Scale x-coordinate of keypoint 1
        int img_y1 = static_cast<int>(y1 * square_dim); // Scale y-coordinate of keypoint 1
        int img_x2 = static_cast<int>(x2 * square_dim); // Scale x-coordinate of keypoint 2
        int img_y2 = static_cast<int>(y2 * square_dim); // Scale y-coordinate of keypoint 2

        // Draw a line between the two keypoints
        cv::line(resized_image, cv::Point(img_x1, img_y1), cv::Point(img_x2, img_y2), cv::Scalar(200, 200, 200), 1);
        // Arguments: image, start point, end point, color, thickness
    }
}
}
int main(int argc, char *argv[]) {
    // Default file paths and settings
    std::string model_file = "assets/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite"; // Path to the MoveNet model file
    std::string video_file = "assets/dancing.mp4"; // Path to the video file
    std::string image_file = ""; // Path to the image file (initially empty)
    bool show_windows = true; // Flag to determine whether to display windows for visualization

    std::map<std::string, std::string> arguments; // Map to store parsed command-line arguments

    // Loop through command-line arguments starting from index 1 (index 0 is the program name)
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]); // Get the current argument

        // Check if the argument starts with "--"
        if (arg.find("--") == 0) {
            size_t equal_sign_pos = arg.find("="); // Find the position of "=" in the argument
            std::string key = arg.substr(0, equal_sign_pos); // Extract the key part of the argument
            std::string value = equal_sign_pos != std::string::npos ? arg.substr(equal_sign_pos + 1) : ""; // Extract the value part of the argument (if exists)

            arguments[key] = value; // Store the key-value pair in the arguments map
        }
    }
// Check if "--model" argument was provided
if (arguments.count("--model")) {
    model_file = arguments["--model"]; // Override default model file with provided value
}

// Check if "--video" argument was provided
if (arguments.count("--video")) {
    video_file = arguments["--video"]; // Override default video file with provided value
}

// Check if "--image" argument was provided
if (arguments.count("--image")) {
    image_file = arguments["--image"]; // Override default image file with provided value
}

// Check if "--no-windows" argument was provided
if (arguments.count("--no-windows")) {
    show_windows = false; // Set show_windows flag to false if "--no-windows" argument is present
}

  // Load the TFLite model from the provided file path
auto model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());

// Check if the model was successfully loaded
if (!model) {
    throw std::runtime_error("Failed to load TFLite model");
}

// Define the resolver for built-in TensorFlow Lite operations
tflite::ops::builtin::BuiltinOpResolver op_resolver;

// Create a TensorFlow Lite interpreter and allocate tensors
std::unique_ptr<tflite::Interpreter> interpreter;
// Invoke the InterpreterBuilder to construct the interpreter
tflite::InterpreterBuilder(*model, op_resolver)(&interpreter);

// Check if interpreter creation and tensor allocation were successful
if (interpreter->AllocateTensors() != kTfLiteOk) {
    throw std::runtime_error("Failed to allocate tensors");
}

// Print interpreter state (for debugging or information purposes)
tflite::PrintInterpreterState(interpreter.get());

// Get details of the input tensor of the interpreter
auto input = interpreter->inputs()[0]; // Get the index of the first input tensor
auto input_tensor = interpreter->tensor(input); // Get the pointer to the input tensor

// Extract input tensor dimensions
auto input_batch_size = input_tensor->dims->data[0]; // Batch size of the input tensor
auto input_height = input_tensor->dims->data[1]; // Height of the input tensor
auto input_width = input_tensor->dims->data[2]; // Width of the input tensor
auto input_channels = input_tensor->dims->data[3]; // Number of channels of the input tensor

// Display the dimensions of the input tensor
std::cout << "The input tensor has the following dimensions: ["
          << input_batch_size << ","  // Display the batch size
          << input_height << ","      // Display the height
          << input_width << ","       // Display the width
          << input_channels << "]"    // Display the number of channels
          << std::endl;

// Get the index of the first output tensor of the interpreter
auto output = interpreter->outputs()[0];

// Get dimensions of the output tensor
auto output_tensor = interpreter->tensor(output); // Get the pointer to the output tensor

// Extract output tensor dimensions
auto dim0 = output_tensor->dims->data[0]; // Dimension 0 of the output tensor
auto dim1 = output_tensor->dims->data[1]; // Dimension 1 of the output tensor
auto dim2 = output_tensor->dims->data[2]; // Dimension 2 of the output tensor
auto dim3 = output_tensor->dims->data[3]; // Dimension 3 of the output tensor

// Display the dimensions of the output tensor
std::cout << "The output tensor has the following dimensions: ["
          << dim0 << ","  // Display dimension 0
          << dim1 << ","  // Display dimension 1
          << dim2 << ","  // Display dimension 2
          << dim3 << "]"  // Display dimension 3
          << std::endl;

// Open the video file using OpenCV for processing
cv::VideoCapture video(video_file);

cv::Mat frame; // Initialize an OpenCV Mat object to hold a frame

// Check if an image file is not provided
if (image_file.empty()) {
    // Check if the video is not opened (if video file is provided)
    if (!video.isOpened()) {
        std::cout << "Can't open the video: " << video_file << std::endl;
        return -1; // Return an error code if video opening fails
    }
}
// If an image file is provided
else {
    frame = cv::imread(image_file); // Read the image file into the 'frame' Mat object
}

// Loop indefinitely for video processing or until a break statement is encountered
while (true) {
    // Check if an image file is not provided (processing video frames)
    if (image_file.empty()) {
        video >> frame; // Read the next frame from the video

        // If the frame is empty (end of video reached or frame read error)
        if (frame.empty()) {
            video.set(cv::CAP_PROP_POS_FRAMES, 0); // Set the video position to the beginning
            continue; // Continue to the next iteration of the loop
        }
    }
    // If an image file is provided, this block won't be executed
    // Get the width and height of the frame (image or video frame)
int image_width = frame.size().width; // Width of the frame
int image_height = frame.size().height; // Height of the frame

// Calculate the dimension of the square area (minimum dimension between width and height)
int square_dim = std::min(image_width, image_height); // Minimum of width and height

// Calculate the delta values for cropping the frame to create a square image
int delta_height = (image_height - square_dim) / 2; // Vertical delta for cropping
int delta_width = (image_width - square_dim) / 2; // Horizontal delta for cropping

// Declare a new Mat object for the resized image
cv::Mat resized_image; // This Mat object will hold the resized image

// Crop and resize the input image
cv::resize(frame(cv::Rect(delta_width, delta_height, square_dim, square_dim)), resized_image, cv::Size(input_width, input_height));

// Copy the resized image data to the input tensor of the interpreter
memcpy(interpreter->typed_input_tensor<unsigned char>(0), resized_image.data, resized_image.total() * resized_image.elemSize());

// Inference
std::chrono::steady_clock::time_point start, end;
start = std::chrono::steady_clock::now();

// Invoke the interpreter to perform inference
if (interpreter->Invoke() != kTfLiteOk) {
    std::cerr << "Inference failed" << std::endl;
    return -1; // Return error code if inference fails
}

end = std::chrono::steady_clock::now();
auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

std::cout << "processing time: " << processing_time << " ms" << std::endl;

// Retrieve output tensor data from the interpreter
float *results = interpreter->typed_output_tensor<float>(0);

// Draw keypoints on the resized image based on the inference results
draw_keypoints(resized_image, results);

if (show_windows) {
    // Show the output image (resized_image) in a window named "Output"
    imshow("Output", resized_image);
} else {
    // If not showing output, run only one frame and then break the loop
    break;
}

// Limit rendering to approximately 30 frames per second
int waitTime = processing_time < 33 ? 33 - processing_time : 1;
if (cv::waitKey(waitTime) >= 0) {
    break; // Break the loop if any key is pressed
}

}
// If no image file is provided (video processing) and video was opened
if (image_file.empty()) {
    video.release(); // Release the video capture object
}

// If show_windows flag is true, close all OpenCV windows
if (show_windows) {
    cv::destroyAllWindows(); // Close all OpenCV windows
}

return 0; // Return 0 indicating successful completion of the program


}
