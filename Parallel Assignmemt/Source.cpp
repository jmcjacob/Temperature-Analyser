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

// Works!
void readData(string file, string location, int year, int month, int day, int time)
{
	string item, line;
	ifstream dataFile(file);
	int count = 0, temp = 0;
	if (dataFile.is_open())
	{
		while (getline(dataFile, line))
		{
			stringstream data = stringstream(line);
			bool write = true;
			string newLocation;
			int newYear, newMonth, newDay, newTime, newTemp;
			while (getline(data, item, ' '))
			{
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
		dataFile.close();
	}
	else
	{
		cout << "Unable to open file" << endl;
		exit(0);
	}
}

// Works (But has an error with setting 0 as the minimum)
int min(cl::Context context, cl::CommandQueue queue, cl::Program program)
{
	vector<int> tempTempTemp = tempTemp;
	size_t localSize = 2; // For now
	size_t paddingSize = tempTempTemp.size() % localSize;

	if (paddingSize)
	{
		std::vector<int> temp(localSize - paddingSize, INT_MAX);
		tempTempTemp.insert(tempTempTemp.end(), temp.begin(), temp.end());
	}

	//std::cout << tempTempTemp << endl;

	size_t inputElements = tempTempTemp.size();
	size_t inputSize = tempTempTemp.size()*sizeof(int);

	std::vector<int> min(1);
	size_t outputSize = sizeof(int);

	cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, inputSize);
	cl::Buffer output(context, CL_MEM_READ_WRITE, outputSize);

	queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, &tempTempTemp[0]);
	queue.enqueueFillBuffer(output, INT_MAX, 0, outputSize);

	cl::Kernel kernel = cl::Kernel(program, "Min");
	kernel.setArg(0, inputBuffer);
	kernel.setArg(1, output);
	kernel.setArg(2, cl::Local(localSize*sizeof(int)));

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(localSize));
	queue.enqueueReadBuffer(output, CL_TRUE, 0, outputSize, &min[0]);

	std::cout << "Minimum: " << (float)min.at(0) / (float)10 << endl;

	return min.at(0);
}

// Works!
int max(cl::Context context, cl::CommandQueue queue, cl::Program program)
{
	vector<int> tempTempTemp = tempTemp;
	size_t localSize = 512; // For now
	size_t paddingSize = tempTempTemp.size() % localSize;

	if (paddingSize)
	{
		std::vector<int> temp(localSize - paddingSize, INT_MIN);
		tempTempTemp.insert(tempTempTemp.end(), temp.begin(), temp.end());
	}

	size_t inputElements = tempTempTemp.size();
	size_t inputSize = tempTempTemp.size()*sizeof(int);

	std::vector<int> max(1);
	size_t outputSize = sizeof(int);

	cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, inputSize);
	cl::Buffer output(context, CL_MEM_READ_WRITE, outputSize);

	queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, &tempTempTemp[0]);
	queue.enqueueFillBuffer(output, INT_MIN, 0, outputSize);

	cl::Kernel kernel = cl::Kernel(program, "Max");
	kernel.setArg(0, inputBuffer);
	kernel.setArg(1, output);
	kernel.setArg(2, cl::Local(localSize*sizeof(int)));

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(localSize));
	queue.enqueueReadBuffer(output, CL_TRUE, 0, outputSize, &max[0]);

	std::cout << "Maximum: " << (float)max.at(0) / (float)10 << endl;

	return max.at(0);
}

// Works!
void average(cl::Context context, cl::CommandQueue queue, cl::Program program)
{
	vector<int> tempTempTemp = tempTemp;
	size_t localSize = 512; // For now
	size_t paddingSize = tempTempTemp.size() % localSize;

	size_t number = tempTempTemp.size();

	if (paddingSize)
	{
		std::vector<int> temp(localSize - paddingSize, 0);
		tempTempTemp.insert(tempTempTemp.end(), temp.begin(), temp.end());
	}

	size_t inputElements = tempTempTemp.size();
	size_t inputSize = tempTempTemp.size()*sizeof(int);

	std::vector<int> output(1);
	size_t outputSize = output.size() * sizeof(int);

	cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, inputSize);
	cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, outputSize);

	queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, &tempTempTemp[0]);
	queue.enqueueFillBuffer(outputBuffer, 0, 0, outputSize);

	cl::Kernel kernel = cl::Kernel(program, "add");
	kernel.setArg(0, inputBuffer);
	kernel.setArg(1, outputBuffer);
	kernel.setArg(2, cl::Local(localSize*sizeof(int)));

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(localSize));
	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, outputSize, &output[0]);

	float answer = ((float)output[0]/ (float)10) / (float)number;
	cout << "Average: " << answer << endl;
}

// Works but is not efficient!
void hisogram(cl::Context context, cl::CommandQueue queue, cl::Program program, int min, int max, int bins)
{
	size_t localSize = bins; 

	vector<int> tempTempTemp = tempTemp;
	size_t paddingSize = tempTempTemp.size() % bins;

	size_t number = tempTempTemp.size();

	if (paddingSize)
	{
		std::vector<int> temp(localSize - paddingSize, 999999);
		tempTempTemp.insert(tempTempTemp.end(), temp.begin(), temp.end());
	}

	size_t inputElements = tempTempTemp.size();
	size_t inputSize = tempTempTemp.size()*sizeof(int);

	std::vector<int> hisogram(bins);
	size_t histoSize = bins * sizeof(int);

	cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, inputSize);
	cl::Buffer histoBuffer(context, CL_MEM_READ_WRITE, histoSize);

	queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, &tempTemp[0]);
	queue.enqueueFillBuffer(histoBuffer, 0, 0, histoSize);
	
	cl::Kernel kernel = cl::Kernel(program, "hist");
	kernel.setArg(0, inputBuffer);
	kernel.setArg(1, cl::Local(localSize*sizeof(int)));
	kernel.setArg(2, histoBuffer);
	kernel.setArg(3, bins);
	kernel.setArg(4, ((float)min/(float)10)*-1);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(bins));
	queue.enqueueReadBuffer(histoBuffer, CL_TRUE, 0, histoSize, &hisogram[0]);

	cout << hisogram << endl;
}

int main(int argc, char **argv)
{
	string location = ""; int bins = 0;
	int platform_id = 0; int device_id = 0;
	int year = 0; int month = 0;
	int day = 0; int time = 0;
	string file = "temp_lincolnshire.txt";

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
		cl::Context context = GetContext(platform_id, device_id);
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
		cl::CommandQueue queue(context);
		cl::Program::Sources sources;
		
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
		
		readData(file, location, year, month, day, time);
		std::cout << "Reading " << tempLocation.size() << " temperatures" << endl;
		
		int minumum = min(context, queue, program);
		int maximum = max(context, queue, program);

		if (bins == 0) { bins = (maximum - minumum)/10; }

		average(context, queue, program);
		hisogram(context, queue, program, minumum, maximum, bins);
	}
	catch (cl::Error err)
	{
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;

	}
}