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
std::vector<float> tempTemp;

void print_help() 
{
	std::cerr << "Application usage:" << std::endl;
	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

void readData(string file)
{
	std::list<string> tempList;
	string item;
	ifstream dataFile(file);
	int count = 0;
	if (dataFile.is_open())
	{
		while (getline (dataFile,item,' ') )
		{
			switch (count)
			{
			case 0:
				tempLocation.push_back(item);
				count++;
				break;
			case 1:
				tempYear.push_back(stoi(item));
				count++;
				break;
			case 2:
				tempMonth.push_back(stoi(item));
				count++;
				break;
			case 3:
				tempDay.push_back(stoi(item));
				count++;
				break;
			case 4:
				tempTime.push_back(stoi(item));
				count++;
				break;
			case 5:
				tempTemp.push_back(stof(item));
				count = 0;
				break;
			default:
				break;
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
	}
	catch (cl::Error err)
	{
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;

	}
}