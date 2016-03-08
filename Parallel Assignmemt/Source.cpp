#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <list>
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

void print_help() 
{
	std::cerr << "Application usage:" << std::endl;
	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

// Works!
void readData(string file)
{
	std::list<string> tempList;
	string item, line;
	ifstream dataFile(file);
	int count = 0, temp = 0;
	if (dataFile.is_open())
	{
		while (getline(dataFile, line))
		{
			stringstream data = stringstream(line);
			while (getline(data, item, ' '))
			{
				switch (count)
				{
				case 0:
					//cout << "Location: " << item << endl;
					tempLocation.push_back(item);
					count++;
					break;
				case 1:
					//cout << "Year: " << item << endl;
					tempYear.push_back(stoi(item));
					count++;
					break;
				case 2:
					//cout << "Month: " << item << endl;
					tempMonth.push_back(stoi(item));
					count++;
					break;
				case 3:
					//cout << "Day: " << item << endl;
					tempDay.push_back(stoi(item));
					count++;
					break;
				case 4:
					//cout << "Time: " << item << endl;
					tempTime.push_back(stoi(item));
					count++;
					break;
				case 5:
					//cout << "Temp: " << item << endl;
					temp = (int)(stof(item)*100);
					tempTemp.push_back(temp);
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

// Works!
void minMax(cl::Context context, cl::CommandQueue queue, cl::Program program)
{
	vector<int> tempTempTemp = tempTemp;
	size_t localSize = 2; // For now
	size_t paddingSize = tempTempTemp.size() % localSize;

	if (paddingSize)
	{
		std::vector<int> temp(localSize - paddingSize, 0.0);
		tempTempTemp.insert(tempTempTemp.end(), temp.begin(), temp.end());
	}

	size_t inputElements = tempTempTemp.size();
	size_t inputSize = tempTempTemp.size()*sizeof(int);

	std::vector<int> minMax(2);
	size_t outputSize = 2 * sizeof(int);

	cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, inputSize);
	cl::Buffer output(context, CL_MEM_READ_WRITE, outputSize);

	queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, &tempTempTemp[0]);
	queue.enqueueFillBuffer(output, 0, 0, outputSize);

	cl::Kernel kernel = cl::Kernel(program, "minMax");
	kernel.setArg(0, inputBuffer);
	kernel.setArg(1, output);
	kernel.setArg(2, cl::Local(localSize*sizeof(int)));
	kernel.setArg(3, cl::Local(localSize*sizeof(int)));

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputElements), cl::NDRange(localSize));
	queue.enqueueReadBuffer(output, CL_TRUE, 0, outputSize, &minMax[0]);

	std::cout << "Minimum: " << (float)minMax.at(0)/100.0 << endl << "Maximum: " << (float)minMax.at(1)/100.0 << endl;
}

// Works!
void average(cl::Context context, cl::CommandQueue queue, cl::Program program)
{
	vector<int> tempTempTemp = tempTemp;
	size_t localSize = 2; // For now
	size_t paddingSize = tempTempTemp.size() % localSize;

	size_t number = tempTempTemp.size();

	if (paddingSize)
	{
		std::vector<int> temp(localSize - paddingSize, 0);
		tempTempTemp.insert(tempTempTemp.end(), temp.begin(), temp.end());
	}

	size_t inputElements = tempTempTemp.size();
	size_t inputSize = tempTempTemp.size()*sizeof(int);

	std::vector<int> output(inputElements);
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

	float answer = ((float)output[0]/ (float)100) / (float)number;
	cout << "Average: " << answer << endl;
}

// Works but is shit
void hisogram(cl::Context context, cl::CommandQueue queue, cl::Program program)
{
	size_t inputElements = tempTemp.size();
	size_t inputSize = inputElements * sizeof(int);

	std::vector<int> hisogram(inputElements);
	size_t histoSize = hisogram.size() * sizeof(int);

	cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, inputSize);
	cl::Buffer histoBuffer(context, CL_MEM_READ_WRITE, histoSize);

	queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, &tempTemp[0]);
	queue.enqueueFillBuffer(histoBuffer, 0, 0, histoSize);

	cl::Kernel kernel = cl::Kernel(program, "hist");
	kernel.setArg(0, inputBuffer);
	kernel.setArg(1, histoBuffer);

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputElements), cl::NullRange);
	queue.enqueueReadBuffer(histoBuffer, CL_TRUE, 0, histoSize, &hisogram[0]);

	cout << hisogram << endl;
}

int main(int argc, char **argv)
{
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) 
	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
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

		readData("temp_lincolnshire_short.txt");
		cout << "Reading " << tempLocation.size() << " temperatures" << endl;
		
		//minMax(context, queue, program);
		//average(context, queue, program);
		hisogram(context, queue, program);
	}
	catch (cl::Error err)
	{
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;

	}
}