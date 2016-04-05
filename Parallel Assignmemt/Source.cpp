#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <iostream>
#include <fstream>
#include <vector>
#include <CL\/cl.h>
#include "Utils.h"

std::vector<string> tempLocation;
std::vector<int> tempYear;
std::vector<int> tempMonth;
std::vector<int> tempDay;
std::vector<int> tempTime;
std::vector<int> tempTemp;

// Reads Data from a given file and will be limited by input.
void readData(string file, string location, int year, int month, int day, int time)
{
	// Specifes varibles for use in method
	string item, line;
	ifstream dataFile(file);
	int count = 0, temp = 0;

	// Checks if file is open
	if (dataFile.is_open())
	{
		// Gets a line from the data file.
		while (getline(dataFile, line))
		{
			// Sets up values to be read into system.
			stringstream data = stringstream(line);
			bool write = true;
			string newLocation;
			int newYear, newMonth, newDay, newTime, newTemp;
			
			// Goes through each item in the line
			while (getline(data, item, ' '))
			{
				// Adds data to vectors is they match the limits
				switch (count)
				{
				case 0:
					newLocation = item;
					if (item != location && location != "") { write = false; }
					count++;
					break;
				case 1:
					newYear = stoi(item);
					if (newYear != year && year != 0) { write = false; }
					count++;
					break;
				case 2:
					newMonth = stoi(item);
					if (newMonth != month && month != 0) { write = false; }
					count++;
					break;
				case 3:
					newDay = stoi(item);
					if (newDay != day && day != 0) { write = false; }
					count++;
					break;
				case 4:
					newTime = stoi(item);
					if (newTime != time && time != 0) { write = false; }
					count++;
					break;
				case 5:
					temp = (int)(stof(item)*10);
					if (write)
					{
						tempLocation.push_back(newLocation);
						tempYear.push_back(newYear);
						tempMonth.push_back(newMonth);
						tempDay.push_back(newDay);
						tempTime.push_back(newTime);
						tempTemp.push_back(temp);
					}
					count = 0;
					break;
				default:
					break;
				}
				
			}
		}

		// Closes data file
		dataFile.close();
	}
	else
	{
		cout << "Unable to open file" << endl;
		exit(0);
	}
}

// Prints and returns the minimum value of the sequence.
int min(cl::Context context, cl::CommandQueue queue, cl::Program program)
{
	// Sets vector, local size and padding size
	vector<int> tempTempTemp = tempTemp;
	size_t localSize = 256; 
	size_t paddingSize = tempTempTemp.size() % localSize;

	// Adds padding to the vector
	if (paddingSize)
	{
		std::vector<int> temp(localSize - paddingSize, INT_MAX);
		tempTempTemp.insert(tempTempTemp.end(), temp.begin(), temp.end());
	}

	// Sets the input elemtents and size of input
	size_t inputElements = tempTempTemp.size();
	size_t inputSize = tempTempTemp.size()*sizeof(int);

	// Sets sizes of output
	std::vector<int> min(1);
	size_t outputSize = sizeof(int);

	// Sets the input and outout buffers
	cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, inputSize);
	cl::Buffer output(context, CL_MEM_READ_WRITE, outputSize);

	// Moves buffer to the GPU
	queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, &tempTempTemp[0]);
	queue.enqueueFillBuffer(output, INT_MAX, 0, outputSize);

	// Sets the kernel and the arguments for the kernel
	cl::Kernel kernel = cl::Kernel(program, "Min");
	kernel.setArg(0, inputBuffer);
	kernel.setArg(1, output);
	kernel.setArg(2, cl::Local(localSize*sizeof(int)));

	// Runs the kernel with a thread for each element of the input with workgroups sized the same as local size
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(localSize));

	// Reads the output from GPU to CPU
	queue.enqueueReadBuffer(output, CL_TRUE, 0, outputSize, &min[0]);

	// Prints and retunes Minimum value
	std::cout << "Minimum: " << (float)min.at(0) / (float)10 << endl;
	return min.at(0);
}

// Prints and returns the maximum value of the sequence.
int max(cl::Context context, cl::CommandQueue queue, cl::Program program)
{
	// Sets vector, local size and padding size
	vector<int> tempTempTemp = tempTemp;
	size_t localSize = 256; // For now
	size_t paddingSize = tempTempTemp.size() % localSize;

	// Adds padding to the vector
	if (paddingSize)
	{
		std::vector<int> temp(localSize - paddingSize, INT_MIN);
		tempTempTemp.insert(tempTempTemp.end(), temp.begin(), temp.end());
	}

	// Sets the input elemtents and size of input
	size_t inputElements = tempTempTemp.size();
	size_t inputSize = tempTempTemp.size()*sizeof(int);

	// Sets sizes of output
	std::vector<int> max(1);
	size_t outputSize = sizeof(int);

	// Sets the input and outout buffers
	cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, inputSize);
	cl::Buffer output(context, CL_MEM_READ_WRITE, outputSize);

	// Moves buffer to the GPU
	queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, &tempTempTemp[0]);
	queue.enqueueFillBuffer(output, INT_MIN, 0, outputSize);

	// Sets the kernel and the arguments for the kernel
	cl::Kernel kernel = cl::Kernel(program, "Max");
	kernel.setArg(0, inputBuffer);
	kernel.setArg(1, output);
	kernel.setArg(2, cl::Local(localSize*sizeof(int)));

	// Runs the kernel with a thread for each element of the input with workgroups sized the same as local size
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(localSize));

	// Reads the output from GPU to CPU
	queue.enqueueReadBuffer(output, CL_TRUE, 0, outputSize, &max[0]);

	// Prints and retunes Maximum value
	std::cout << "Maximum: " << (float)max.at(0) / (float)10 << endl;
	return max.at(0);
}

// Prints the average value of the sequence.
void average(cl::Context context, cl::CommandQueue queue, cl::Program program)
{
	// Sets vector, local size and padding size
	vector<int> tempTempTemp = tempTemp;
	size_t localSize = 256; 
	size_t paddingSize = tempTempTemp.size() % localSize;

	// Gets size of vector before padding to get accurate average
	size_t number = tempTempTemp.size();

	// Adds padding to the vector
	if (paddingSize)
	{
		std::vector<int> temp(localSize - paddingSize, 0);
		tempTempTemp.insert(tempTempTemp.end(), temp.begin(), temp.end());
	}

	// Sets the input elemtents and size of input
	size_t inputElements = tempTempTemp.size();
	size_t inputSize = tempTempTemp.size()*sizeof(int);

	// Sets sizes of output
	std::vector<int> output(1);
	size_t outputSize = output.size() * sizeof(int);

	// Sets the input and outout buffers
	cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, inputSize);
	cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, outputSize);

	// Moves buffer to the GPU
	queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, &tempTempTemp[0]);
	queue.enqueueFillBuffer(outputBuffer, 0, 0, outputSize);

	// Sets the kernel and the arguments for the kernel
	cl::Kernel kernel = cl::Kernel(program, "add");
	kernel.setArg(0, inputBuffer);
	kernel.setArg(1, outputBuffer);
	kernel.setArg(2, cl::Local(localSize*sizeof(int)));

	// Runs the kernel with a thread for each element of the input with workgroups sized the same as local size
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(localSize));

	// Reads the output from GPU to CPU
	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, outputSize, &output[0]);

	// Calculates and prints Average
	float answer = ((float)output[0]/ (float)10) / (float)number;
	cout << "Average: " << answer << endl;
}

// Prints the histogram with the specified number of bins from the sequence.
void hisogram(cl::Context context, cl::CommandQueue queue, cl::Program program, int min, int max, int bins)
{
	// Sets vector, local size and padding size
	vector<int> tempTempTemp = tempTemp;
	size_t localSize = 258;
	size_t paddingSize = tempTempTemp.size() % localSize;

	// Adds padding to the vector
	if (paddingSize)
	{
		std::vector<int> temp(localSize - paddingSize, 999999);
		tempTempTemp.insert(tempTempTemp.end(), temp.begin(), temp.end());
	}

	// Sets the input elemtents and size of input
	size_t inputElements = tempTempTemp.size();
	size_t inputSize = tempTempTemp.size()*sizeof(int);

	// Sets sizes of output to number of bins
	std::vector<int> hisogram(bins);
	size_t histoSize = bins * sizeof(int);

	// Sets the input and outout buffers
	cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, inputSize);
	cl::Buffer histoBuffer(context, CL_MEM_READ_WRITE, histoSize);

	// Moves buffer to the GPU
	queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, &tempTempTemp[0]);
	queue.enqueueFillBuffer(histoBuffer, 0, 0, histoSize);
	
	// Sets the kernel and the arguments for the kernel
	cl::Kernel kernel = cl::Kernel(program, "hist");
	kernel.setArg(0, inputBuffer);
	kernel.setArg(1, cl::Local(histoSize));
	kernel.setArg(2, histoBuffer);
	kernel.setArg(3, bins);
	kernel.setArg(4, max);
	kernel.setArg(5, min);

	// Runs the kernel with a thread for each element of the input with workgroups sized the same as local size
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(localSize));

	// Reads the output from GPU to CPU
	queue.enqueueReadBuffer(histoBuffer, CL_TRUE, 0, histoSize, &hisogram[0]);

	// Prints the histogram
	cout << hisogram << endl;
}

int main(int argc, char **argv)
{
	// Sets defult values
	string location = ""; int bins = 0;
	int platform_id = 0; int device_id = 0;
	int year = 0; int month = 0;
	int day = 0; int time = 0;
	string file = "temp_lincolnshire.txt";

	// Reads for inputs to the application
	for (int i = 1; i < argc; i++) 
	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "--Location") == 0) && (i < (argc - 1))) { location = argv[++i]; }
		else if ((strcmp(argv[i], "--Year") == 0) && (i < (argc - 1))) { year = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "--Month") == 0) && (i < (argc - 1))) { month = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "--Day") == 0) && (i < (argc - 1))) { day = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "--Time") == 0) && (i < (argc - 1))) { time = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "--Bins") == 0) && (i < (argc - 1))) { bins = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "--Short") == 0) && (i < (argc - 1))) { if (atoi(argv[++i]) == 1) { file = "temp_lincolnshire_short.txt"; } }
	}

	try
	{
		// Sets the context of the GPU
		cl::Context context = GetContext(platform_id, device_id);
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
		cl::CommandQueue queue(context);
		cl::Program::Sources sources;
		
		// Adds sources and builds it
		AddSources(sources, "kernels.cl");
		cl::Program program(context, sources);
		try
		{
			program.build();
		}
		catch (const cl::Error& err)
		{
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		
		// Reads the data and prints the number of elements its read
		readData(file, location, year, month, day, time);
		std::cout << "Reading " << tempLocation.size() << " temperatures" << endl;
		
		// Gets min and max
		int minumum = min(context, queue, program);
		int maximum = max(context, queue, program);

		// Get number of bins
		if (bins == 0) { bins = 10; }

		// Gets average and hsitogram
		average(context, queue, program);
		hisogram(context, queue, program, minumum, maximum, bins);
	}
	catch (cl::Error err)
	{
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;

	}
}