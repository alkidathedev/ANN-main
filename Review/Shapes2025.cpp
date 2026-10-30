#define _CRT_SECURE_NO_WARNINGS
// name of the project: Shapes2025
#include <iostream>
#include <fstream>
#include <string>
#include <io.h>
#include <dos.h>
#include <conio.h>
#include <stdlib.h>
#include <sstream>
#include <stdio.h>
#include <iomanip>
#include <istream>
#include <math.h>

using namespace std;

// declare a class by the name 'Shapes2025'
class Shapes2025 {

	// the point is to assign to 'n1' and 'n2' 
	// the correct values from the command console
	int n1; // matrix size x (number of pixels along the x direction)
	int n2; // matrix size y (number of pixels along the y direction)

    char NoiseLabel[100];
	// declare the class methods
	// and embed the methods body
	// in the class specification
public:

	// declare a method that returns the number
	// of pixels of the image along the x direction
	int getNofPixelsX(void) { return this->n1; };

	// declare a method that returns the number
	// of pixels of the image along the y direction
	int getNofPixelsY(void) { return this->n2; };

	// declare a method that sets the number
	// of pixels of the image along the x direction
	void setNofPixelsX(int x) { this->n1 = x; };

	// declare a method that sets the number 
	// of pixels of the image along the y direction
	void setNofPixelsY(int y) { this->n2 = y; };

public:
	// declare a structure ' data ' that defines the
	// pointers to pointers to the image
	struct data {	

		double** Square; // declare the pointer to pointer to the matrix (image)

		double** Rectangle; // declare the pointer to pointer to the matrix (image)

	}*pointer; // pointer to the element of the structure 'data'
	// the pointer points to the memory address of the
	// pointers to pointers 

public:

	// constructor of the class 'Shapes2025'
	Shapes2025(int x, int y) : n1(x), n2(y) { pointer = 0; };

	// declare the prototype of the 
	// function 'allocateData'
	// the function belongs to the 
	// sets of methods of the class 'Shapes2025'
	void allocateData();

	// declare the prototype of the 
	// function 'save'
	// the function belongs to the 
	// sets of methods of the class 'Shapes2025'
	void save(char NoiseLabel[], int n1, int n2);

	// destructor of the class 'Shapes2025'
	~Shapes2025() { }

};

void Shapes2025::allocateData() { // allocate data


	// (1) allocate struct 'data' (begin)
	// define a pointer by the name 'pointer'
	// and assign to it the instance of the
	// structure 'data' (the instance is a
	// memory address
	pointer = new data;

	// assign to the pointer to a pointer 'Signal' (pointer->Signal)
	// the instance of the memory address of the pointer to pointer
	// *[this->n2]. Practically define the memory address of the 
	// rows of the matrix containing the image 'Signal'.
	

	pointer->Square = new double* [this->n1];

	pointer->Rectangle = new double* [this->n1];


	for (int v = 0; v < this->n1; v++) { // (1)
		// at each iteration of the for loop
		// assign to the pointer 'Signal[v]' the instance 
		// of the memory address of the pointer [this-n1].
		// Practically define the memory address of the 
		// columns of the matrices containing the image:
		// 'Signal.

		
		pointer->Square[v] = new double[this->n2];

		pointer->Rectangle[v] = new double[this->n2];

	} // (1) allocate struct 'data' (end)


   // (2) initialize (begin)
	for (int v = 0; v < this->n1; v++) { // (a)

		for (int f = 0; f < this->n2; f++) { // (b)

			// at each iteration of the two for loops
			// initializes the value of the pixel of
			// the image to zero. This is done for the
			// image 'Signal'.
			

			pointer->Square[v][f] = (double)0.0;

			pointer->Rectangle[v][f] = (double)0.0;

		} //(b)

	} //(a)
   // (2) initialize (end)

} // allocate data


void Shapes2025::save(char NoiseLabel[], int n1, int n2) { // saveImage (begins)

	FILE* savedata;
	char outputFile[128];

	sprintf(outputFile, "%s%d%s%d%s%s%s", "Square_", n1, "x", n2, "_", NoiseLabel, ".img");
	if ((savedata = fopen(outputFile, "wb+")) == NULL)
	{

		std::cout << "Cannot open output file, Now Exit..." << endl;

	}
	else { // (save)

		for (int v = 0; v < this->n1; v++) {

			for (int f = 0; f < this->n2; f++) {

				fwrite(&pointer->Square[v][f], sizeof(double), 1, savedata);

			}
		}

		fclose(savedata);

	} // (save)

	sprintf(outputFile, "%s%d%s%d%s%s%s", "Rectangle_", n1, "x", n2, "_", NoiseLabel, ".img");
	if ((savedata = fopen(outputFile, "wb+")) == NULL)
	{

		std::cout << "Cannot open output file, Now Exit..." << endl;

	}
	else { // (save)

		for (int v = 0; v < this->n1; v++) {

			for (int f = 0; f < this->n2; f++) {

				fwrite(&pointer->Rectangle[v][f], sizeof(double), 1, savedata);

			}
		}

		fclose(savedata);

	} // (save)
} // saveImage (ends)

// read the input parameters from the
// console command line
int main(int argc, char* argv[]) {

	// assign to the char string 'outputFile'
	// the value "Shapes2025.log"
	char outputFile[128] = "Shapes2025.log";

	// declare a pointer to a file by the
	// name 'savedata'
	FILE* savedata;

	// tell the user of the list of input parameters necessary to tun the program:
	if (argc < 7) {
		std::cout << endl;
		std::cout << "Please enter the number of pixels along the X direction (integer)" << endl;
		std::cout << "Please enter the number of pixels along the Y direction (integer)" << endl;
		std::cout << "Please enter the amplitude of pixel inside the shapes:" << endl;
		std::cout << "Please choose in the range [1.0, 1000.0]" << endl;
		std::cout << "Please enter the X size of the Rectangle (integer)" << endl;
		std::cout << "Please enter the Y size of the Rectangle (integer)"  << endl;
		std::cout << "Please enter edge of the Square (integer)" << endl;
		std::cout << endl;
		exit(0);
	}

	else { // run the program (begin)

		// opens the log file which name is 
		// contained in 'outputFile', opens 
		// in write mode
		if ((savedata = fopen(outputFile, "w")) == NULL)
		{
			// alert the user of possible failure in opening the file
			std::cout << "Cannot open output file, Now Exit..." << endl;
			exit(0);

		}
		else { // processing (begin)

			// reads from the command console the
			// value of the image size (number of rows
			// and number of columns)
			int n1 = atoi(argv[1]);
			int n2 = atoi(argv[2]);
			double amplitude = atof(argv[3]);

			int xSize = atoi(argv[4]);
			int ySize = atoi(argv[5]);
			int edge = atoi(argv[6]);
			

			if ((double)amplitude < 1.0 || (double)amplitude > 1000.0)
			{

				std::cout << "Please enter the amplitude of the waves (double) in the range [1.0, 1000.0]" << endl;
				exit(0);
			}

			if ((int)xSize < 1 || (double)xSize > (int)n1/2)
			{

				std::cout << "The X size of the Rectangle (integer) needs to be: " << endl;
				std::cout << "greater than 1 and less than: " << (int)n1/2 << endl;
				exit(0);
			}

			if ((int)ySize < 2 || (double)ySize > (int)n2/2)
			{

				std::cout << "The Y size of the Rectangle (integer) needs to be: " << endl;
				std::cout << "greater than 2 and less than: " << (int)n2/2 << endl;
				exit(0);
			}

			if ((int)edge < 2 || edge > (double)(n1/2) || edge > (double)(n2/2))
			{

				std::cout << "The edge of the Square (integer) needs to be: " << endl;
				std::cout << "greater than 2 and less than the smallest between " << endl;
				std::cout << (int)n1/2 << " and " << (int)n2/2 << endl;
				exit(0);
			}
		
			// inform the user of the image size 
			// (number of rows and number of columns
			// of the matrix containing the image)
			std::cout << endl;
			std::cout << "The number of pixels along the X direction is: " << atoi(argv[1]) << endl;
			std::cout << "The number of pixels along the Y direction is: " << atoi(argv[2]) << endl;
			std::cout << "The amplitude of the pixel brightness is: " << atof(argv[3]) << endl;
			std::cout << "The X size of the Rectangle is: " << atoi(argv[4]) << endl;
			std::cout << "The Y size of the Rectangle is: " << atoi(argv[5]) << endl;
			std::cout << "The edge of the Square is: " << atoi(argv[6]) << endl;
		

			// save into the log file the image size 
			// (number of rows and number of columns
			// of the matrix containing the image)
			fprintf(savedata, "%s%d\n", "The number of pixels along the X direction is: ", n1);
			fprintf(savedata, "%s%d\n", "The number of pixels along the Y direction is: ", n2);
			fprintf(savedata, "%s%lf\n", "The amplitude of the pixel brightness is: ", atof(argv[3]));
			fprintf(savedata, "%s%d\n", "The X size of the Rectangle is: ", atoi(argv[4]));
			fprintf(savedata, "%s%d\n", "The Y size of the Rectangle is: " , atoi(argv[5]));
			fprintf(savedata, "%s%d\n", "The edge of the Square is: ", atoi(argv[6]));
			

			// call to the constructor so to create
			// an object. The data type of the object 
			// is 'Shapes2025'
			Shapes2025 Signal(n2, n1);

			// the object
			// sends a message (invokes)
			// to the method 'allocateData()'
			Signal.allocateData();

			std::cout << "Now Processing..." << endl;
		
			char TheLabel[100];
			int f = 0;
			int counter = 0;
	
					int xEdge = ((int) xSize);
					int yEdge = ((int) ySize);

					std::cout << "Rectangle xEdge = " << xEdge << endl;
					std::cout << "Rectangle yEdge = " << yEdge << endl;
					std::cout << "Square Edge = " << edge << endl;

					    //////////////////////Along the Y Axis/////////////////////////
						for (f = 1; f < (n2/2)-yEdge; f += yEdge) { // along negative y
							counter++;
						
							   int i1_leftR  = n1/2 - xEdge/2; 
							   int i1_rightR = n1/2 + xEdge/2;

							   int i2_leftR  = f + n2/2 - yEdge/2; 
							   int i2_rightR = f + n2/2 + yEdge/2;

							   if (i2_leftR >= 0 && i2_leftR < n2) // if (1)
							    { 
							
									if (i2_rightR >= 0 && i2_rightR < n2) // if (2)
							        { 

													   
							   for (int i1 = i2_leftR; i1 <= i2_rightR; i1++) { 

							     for (int i2 = i1_leftR; i2 <= i1_rightR; i2++) {
			
								Signal.pointer->Rectangle[i1][i2] = (double)amplitude;
					                      
								}
							}				   
							  
									} // if (1)
							   } // if (2)

							   
							   int i1_leftS  = n1/2 - edge/2; 
							   int i1_rightS = n1/2 + edge/2;

							   int i2_leftS  = f + n2/2 - edge/2; 
							   int i2_rightS = f + n2/2 + edge/2;

							   if (i2_leftS >= 0 && i2_leftS < n2) // if (1)
							    { 
							
									if (i2_rightS >= 0 && i2_rightS < n2) // if (2)
							        { 

													   
							   for (int i1 = i2_leftS; i1 <= i2_rightS; i1++) { 

							     for (int i2 = i1_leftS; i2 <= i1_rightS; i2++) {
			
								Signal.pointer->Square[i1][i2] = (double)amplitude;
					                      
								}
							}				   
							  
									} // if (1)
							   } // if (2)

							    sprintf(TheLabel, "%d", counter);
								Signal.save(TheLabel, n1, n2);

								for (int s = 0; s < n2; s++) { 

							        for (int p = 0; p < n1; p++) {
			
								     Signal.pointer->Rectangle[s][p] = (double)0.0;
					                 Signal.pointer->Square[s][p] = (double)0.0;        
								}
							}	

						} // along negative y

						////////////////////////////Along the Y axis///////////////////////
						for (f = yEdge; f < (n2/2)-yEdge; f += yEdge) { // along positive y
							counter++;
						
							   int i1_leftR  = n1/2 - xEdge/2; 
							   int i1_rightR = n1/2 + xEdge/2;

							   int i2_leftR  = -f + n2/2 - yEdge/2; 
							   int i2_rightR = -f + n2/2 + yEdge/2;

							   if (i2_leftR >= 0 && i2_leftR < n2) // if (1)
							    { 
							
									if (i2_rightR >= 0 && i2_rightR < n2) // if (2)
							        { 

													   
							   for (int i1 = i2_leftR; i1 <= i2_rightR; i1++) { 

							     for (int i2 = i1_leftR; i2 <= i1_rightR; i2++) {
			
								Signal.pointer->Rectangle[i1][i2] = (double)amplitude;
					                      
								}
							}				   
							  
									} // if (1)
							   } // if (2)

							   
							   int i1_leftS  = n1/2 - edge/2; 
							   int i1_rightS = n1/2 + edge/2;

							   int i2_leftS  = -f + n2/2 - edge/2; 
							   int i2_rightS = -f + n2/2 + edge/2;

							   if (i2_leftS >= 0 && i2_leftS < n2) // if (1)
							    { 
							
									if (i2_rightS >= 0 && i2_rightS < n2) // if (2)
							        { 

													   
							   for (int i1 = i2_leftS; i1 <= i2_rightS; i1++) { 

							     for (int i2 = i1_leftS; i2 <= i1_rightS; i2++) {
			
								Signal.pointer->Square[i1][i2] = (double)amplitude;
					                      
								}
							}				   
							  
									} // if (1)
							   } // if (2)

							    sprintf(TheLabel, "%d", counter);
								Signal.save(TheLabel, n1, n2);

								for (int s = 0; s < n2; s++) { 

							        for (int p = 0; p < n1; p++) {
			
								     Signal.pointer->Rectangle[s][p] = (double)0.0;
					                 Signal.pointer->Square[s][p] = (double)0.0;        
								}
							}	

						} // along positive y
						/////////////////////////Along the Y axis//////////////////////////////

						////////////////////////Along the X axis///////////////////////////////
							for (f = xEdge; f < (n1/2)-xEdge; f += xEdge) { // along negative x
							counter++;
						
							   int i1_leftR  = f + n1/2 - xEdge/2; 
							   int i1_rightR = f + n1/2 + xEdge/2;

							   int i2_leftR  = n2/2 - yEdge/2; 
							   int i2_rightR = n2/2 + yEdge/2;

							   if (i2_leftR >= 0 && i2_leftR < n2) // if (1)
							    { 
							
									if (i2_rightR >= 0 && i2_rightR < n2) // if (2)
							        { 

													   
							   for (int i1 = i2_leftR; i1 <= i2_rightR; i1++) { 

							     for (int i2 = i1_leftR; i2 <= i1_rightR; i2++) {
			
								Signal.pointer->Rectangle[i1][i2] = (double)amplitude;
					                      
								}
							}				   
							  
									} // if (1)
							   } // if (2)

							   
							   int i1_leftS  = f + n1/2 - edge/2; 
							   int i1_rightS = f + n1/2 + edge/2;

							   int i2_leftS  = n2/2 - edge/2; 
							   int i2_rightS = n2/2 + edge/2;

							   if (i2_leftS >= 0 && i2_leftS < n2) // if (1)
							    { 
							
									if (i2_rightS >= 0 && i2_rightS < n2) // if (2)
							        { 

													   
							   for (int i1 = i2_leftS; i1 <= i2_rightS; i1++) { 

							     for (int i2 = i1_leftS; i2 <= i1_rightS; i2++) {
			
								Signal.pointer->Square[i1][i2] = (double)amplitude;
					                      
								}
							}				   
							  
									} // if (1)
							   } // if (2)

							    sprintf(TheLabel, "%d", counter);
								Signal.save(TheLabel, n1, n2);

								for (int s = 0; s < n2; s++) { 

							        for (int p = 0; p < n1; p++) {
			
								     Signal.pointer->Rectangle[s][p] = (double)0.0;
					                 Signal.pointer->Square[s][p] = (double)0.0;        
								}
							}	

						} // along negative x
                            ///////////////////////////////Along the X axis////////////////////
							for (f = xEdge; f < (n1/2)-xEdge; f += xEdge) { // along positive x
							counter++;
						
							   int i1_leftR  = -f + n1/2 - xEdge/2; 
							   int i1_rightR = -f + n1/2 + xEdge/2;

							   int i2_leftR  = n2/2 - yEdge/2; 
							   int i2_rightR = n2/2 + yEdge/2;

							   if (i2_leftR >= 0 && i2_leftR < n2) // if (1)
							    { 
							
									if (i2_rightR >= 0 && i2_rightR < n2) // if (2)
							        { 

													   
							   for (int i1 = i2_leftR; i1 <= i2_rightR; i1++) { 

							     for (int i2 = i1_leftR; i2 <= i1_rightR; i2++) {
			
								Signal.pointer->Rectangle[i1][i2] = (double)amplitude;
					                      
								}
							}				   
							  
									} // if (1)
							   } // if (2)

							   
							   int i1_leftS  = -f + n1/2 - edge/2; 
							   int i1_rightS = -f + n1/2 + edge/2;

							   int i2_leftS  = n2/2 - edge/2; 
							   int i2_rightS = n2/2 + edge/2;

							   if (i2_leftS >= 0 && i2_leftS < n2) // if (1)
							    { 
							
									if (i2_rightS >= 0 && i2_rightS < n2) // if (2)
							        { 

													   
							   for (int i1 = i2_leftS; i1 <= i2_rightS; i1++) { 

							     for (int i2 = i1_leftS; i2 <= i1_rightS; i2++) {
			
								Signal.pointer->Square[i1][i2] = (double)amplitude;
					                      
								}
							}				   
							  
									} // if (1)
							   } // if (2)

							    sprintf(TheLabel, "%d", counter);
								Signal.save(TheLabel, n1, n2);

								for (int s = 0; s < n2; s++) { 

							        for (int p = 0; p < n1; p++) {
			
								     Signal.pointer->Rectangle[s][p] = (double)0.0;
					                 Signal.pointer->Square[s][p] = (double)0.0;        
								}
							}	

						} // along positive x
                        


						 ///////////////////////////Along the Diagonal//////////////////////
						for (f = 1; f < (n2 / 2) - yEdge; f += yEdge) {
							counter++;

							int i1_leftR = (int)((n1 / 2 - xEdge / 2) * ((double)sqrt((double)2.0) / 2.0));
							int i1_rightR = (int)((n1 / 2 + xEdge / 2) * ((double)sqrt((double)2.0) / 2.0));

							int i2_leftR = (int)((f + n2 / 2 - yEdge / 2) * ((double)sqrt((double)2.0) / 2.0));
							int i2_rightR = (int)((f + n2 / 2 + yEdge / 2) * ((double)sqrt((double)2.0) / 2.0));

							if (i2_leftR >= 0 && i2_leftR < n2) // if (1)
							{

								if (i2_rightR >= 0 && i2_rightR < n2) // if (2)
								{


									for (int i1 = i2_leftR; i1 <= i2_rightR; i1++) {

										for (int i2 = i1_leftR; i2 <= i1_rightR; i2++) {

											Signal.pointer->Rectangle[i1][i2] = (double)amplitude;

										}
									}

								} // if (1)
							} // if (2)


							int i1_leftS = (int)((n1 / 2 - edge / 2) * ((double)sqrt((double)2.0) / 2.0));
							int i1_rightS = (int)((n1 / 2 + edge / 2) * ((double)sqrt((double)2.0) / 2.0));

							int i2_leftS = (int)((f + n2 / 2 - edge / 2) * ((double)sqrt((double)2.0) / 2.0));
							int i2_rightS = (int)((f + n2 / 2 + edge / 2) * ((double)sqrt((double)2.0) / 2.0));

							if (i2_leftS >= 0 && i2_leftS < n2) // if (1)
							{

								if (i2_rightS >= 0 && i2_rightS < n2) // if (2)
								{


									for (int i1 = i2_leftS; i1 <= i2_rightS; i1++) {

										for (int i2 = i1_leftS; i2 <= i1_rightS; i2++) {

											Signal.pointer->Square[i1][i2] = (double)amplitude;

										}
									}

								} // if (1)
							} // if (2)

							sprintf(TheLabel, "%d", counter);
							Signal.save(TheLabel, n1, n2);

							for (int s = 0; s < n2; s++) {

								for (int p = 0; p < n1; p++) {

									Signal.pointer->Rectangle[s][p] = (double)0.0;
									Signal.pointer->Square[s][p] = (double)0.0;
								}
							}

						}
							/////////////////////Along the Diagonal////////////////////////////////
							for (f = yEdge; f < (n2 / 2) - yEdge; f += yEdge) { 
								counter++;

								int i1_leftR = (int)((n1 / 2 - xEdge / 2) * ((double)sqrt((double)2.0) / 2.0));
								int i1_rightR = (int)((n1 / 2 + xEdge / 2) * ((double)sqrt((double)2.0) / 2.0));

								int i2_leftR = (int)((-f + n2 / 2 - yEdge / 2) * ((double)sqrt((double)2.0) / 2.0));
								int i2_rightR = (int)((-f + n2 / 2 + yEdge / 2) * ((double)sqrt((double)2.0) / 2.0));

								if (i2_leftR >= 0 && i2_leftR < n2) // if (1)
								{

									if (i2_rightR >= 0 && i2_rightR < n2) // if (2)
									{


										for (int i1 = i2_leftR; i1 <= i2_rightR; i1++) {

											for (int i2 = i1_leftR; i2 <= i1_rightR; i2++) {

												Signal.pointer->Rectangle[i1][i2] = (double)amplitude;

											}
										}

									} // if (1)
								} // if (2)


								int i1_leftS = (int)((n1 / 2 - edge / 2) * ((double)sqrt((double)2.0) / 2.0));
								int i1_rightS = (int)((n1 / 2 + edge / 2) * ((double)sqrt((double)2.0) / 2.0));

								int i2_leftS = (int)((-f + n2 / 2 - edge / 2) * ((double)sqrt((double)2.0) / 2.0));
								int i2_rightS = (int)((-f + n2 / 2 + edge / 2) * ((double)sqrt((double)2.0) / 2.0));

								if (i2_leftS >= 0 && i2_leftS < n2) // if (1)
								{

									if (i2_rightS >= 0 && i2_rightS < n2) // if (2)
									{


										for (int i1 = i2_leftS; i1 <= i2_rightS; i1++) {

											for (int i2 = i1_leftS; i2 <= i1_rightS; i2++) {

												Signal.pointer->Square[i1][i2] = (double)amplitude;

											}
										}

									} // if (1)
								} // if (2)

								sprintf(TheLabel, "%d", counter);
								Signal.save(TheLabel, n1, n2);

								for (int s = 0; s < n2; s++) {

									for (int p = 0; p < n1; p++) {

										Signal.pointer->Rectangle[s][p] = (double)0.0;
										Signal.pointer->Square[s][p] = (double)0.0;
									}
								}

							}
						
							///////////////////////////////////Along the diagonal////////////////////////
							for (f = xEdge; f < (n1 / 2) - xEdge; f += xEdge) {
								counter++;

								int i1_leftR = (int)((f + n1 / 2 - xEdge / 2) * ((double)sqrt((double)2.0) / 2.0));
								int i1_rightR = (int)((f + n1 / 2 + xEdge / 2) * ((double)sqrt((double)2.0) / 2.0));

								int i2_leftR = (int)((n2 / 2 - yEdge / 2) * ((double)sqrt((double)2.0) / 2.0));
								int i2_rightR = (int)((n2 / 2 + yEdge / 2) * ((double)sqrt((double)2.0) / 2.0));

								if (i2_leftR >= 0 && i2_leftR < n2) // if (1)
								{

									if (i2_rightR >= 0 && i2_rightR < n2) // if (2)
									{


										for (int i1 = i2_leftR; i1 <= i2_rightR; i1++) {

											for (int i2 = i1_leftR; i2 <= i1_rightR; i2++) {

												Signal.pointer->Rectangle[i1][i2] = (double)amplitude;

											}
										}

									} // if (1)
								} // if (2)


								int i1_leftS = (int)((f + n1 / 2 - edge / 2) * ((double)sqrt((double)2.0) / 2.0));
								int i1_rightS = (int)((f + n1 / 2 + edge / 2) * ((double)sqrt((double)2.0) / 2.0));

								int i2_leftS = (int)((n2 / 2 - edge / 2) * ((double)sqrt((double)2.0) / 2.0));
								int i2_rightS = (int)((n2 / 2 + edge / 2) * ((double)sqrt((double)2.0) / 2.0));

								if (i2_leftS >= 0 && i2_leftS < n2) // if (1)
								{

									if (i2_rightS >= 0 && i2_rightS < n2) // if (2)
									{


										for (int i1 = i2_leftS; i1 <= i2_rightS; i1++) {

											for (int i2 = i1_leftS; i2 <= i1_rightS; i2++) {

												Signal.pointer->Square[i1][i2] = (double)amplitude;

											}
										}

									} // if (1)
								} // if (2)

								sprintf(TheLabel, "%d", counter);
								Signal.save(TheLabel, n1, n2);

								for (int s = 0; s < n2; s++) {

									for (int p = 0; p < n1; p++) {

										Signal.pointer->Rectangle[s][p] = (double)0.0;
										Signal.pointer->Square[s][p] = (double)0.0;
									}
								}

							}

							/////////////////////////Along the Diagonal////////////////////////////
							for (f = xEdge; f < (n1 / 2) - xEdge; f += xEdge) {
								counter++;

								int i1_leftR = (int)((-f + n1 / 2 - xEdge / 2) * ((double)sqrt((double)2.0) / 2.0));
								int i1_rightR = (int)((-f + n1 / 2 + xEdge / 2) * ((double)sqrt((double)2.0) / 2.0));

								int i2_leftR = (int)((n2 / 2 - yEdge / 2) * ((double)sqrt((double)2.0) / 2.0));
								int i2_rightR = (int)((n2 / 2 + yEdge / 2) * ((double)sqrt((double)2.0) / 2.0));

								if (i2_leftR >= 0 && i2_leftR < n2) // if (1)
								{

									if (i2_rightR >= 0 && i2_rightR < n2) // if (2)
									{


										for (int i1 = i2_leftR; i1 <= i2_rightR; i1++) {

											for (int i2 = i1_leftR; i2 <= i1_rightR; i2++) {

												Signal.pointer->Rectangle[i1][i2] = (double)amplitude;

											}
										}

									} // if (1)
								} // if (2)


								int i1_leftS = (int)((-f + n1 / 2 - edge / 2) * ((double)sqrt((double)2.0) / 2.0));
								int i1_rightS = (int)((-f + n1 / 2 + edge / 2) * ((double)sqrt((double)2.0) / 2.0));

								int i2_leftS = (int)((n2 / 2 - edge / 2) * ((double)sqrt((double)2.0) / 2.0));
								int i2_rightS = (int)((n2 / 2 + edge / 2) * ((double)sqrt((double)2.0) / 2.0));

								if (i2_leftS >= 0 && i2_leftS < n2) // if (1)
								{

									if (i2_rightS >= 0 && i2_rightS < n2) // if (2)
									{


										for (int i1 = i2_leftS; i1 <= i2_rightS; i1++) {

											for (int i2 = i1_leftS; i2 <= i1_rightS; i2++) {

												Signal.pointer->Square[i1][i2] = (double)amplitude;

											}
										}

									} // if (1)
								} // if (2)

								sprintf(TheLabel, "%d", counter);
								Signal.save(TheLabel, n1, n2);

								for (int s = 0; s < n2; s++) {

									for (int p = 0; p < n1; p++) {

										Signal.pointer->Rectangle[s][p] = (double)0.0;
										Signal.pointer->Square[s][p] = (double)0.0;
									}
								}

							}
							////////////////////////////////Along the Diagonal////////////////////////////////

			std::cout << "Image Data Calculated" << endl;
			// calculate Signals (ends)

			// alert the user of the end of the program
			std::cout << "End of Computation..." << endl;
			std::cout << endl;

			// save to log file
			fprintf(savedata, "%s\n", "End of Computation...");
			fprintf(savedata, "\n");

			fclose(savedata);
			// delete the memory address
			// allocated to the images
			// do this by the command 'delete'
			// applied to the object which
			// invokes the pointer to the data
			// structure 'data' containing the image
			// 'Signal'
			delete Signal.pointer;
			// the object invokes the
			// class destructor
			Signal.~Shapes2025();

		} // processing (end)

	} // run the program (end)

	// ANSI C requires the 'main'
	// function returning a value: 
	// zero in this case
	return 0;
} // end of main 