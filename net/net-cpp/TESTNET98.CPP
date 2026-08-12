
       #include <stdio.h>
       #include <alloc.h>
       #include <string.h>
       #include <dos.h>
       #include <stdlib.h>
       #include <conio.h>
       #include <math.h>
       #include <ctype.h>




  struct learning_set

     {
	unsigned int  T ;   //unsigned  int//
	int INP ;
	int HID ;
	int OUT ;

      };

      struct learning_set * set;

      char classNames[10][100];

      int  set_learning_variables(char path_store[80]);

      float huge * first_layer_connections_taker( int p, int q );

      float huge * second_layer_connections_taker( int r, int v);

      float far * biggest_range_taker( int r );

      float far * smallest_range_taker( int r );

      float far * pattern_normaliser(char filename[128], float far *p1 ,float far *
					p2 , int x);

      float     normalise(float value, float min, float max);

      char * take_a_filename( char current_dir[80],unsigned int p);

      float sigmoid(float input);

      float far * output_nodes_producer( float huge * l1 , float far * l2 ,
				      int p ,int r);





	void main(void){


	int    k6;
	unsigned int j,t;

	char string[80];
	char * name;
	char filename[128];

	FILE *stream;



	float huge * input_layer;
	float huge * hidden_layer;

	float far  * upper;
	float far  * lower;
	float far  * pattern;

	float far  *  hidden_activator;
	float far  *  output_activator;


	   char *lena;
	   char *aigo;
	   char *goulp;

	   goulp="NETWORK RESPONSE:";




     printf("INSERT TEST DIRECTORY [drive:\\directory\\*.*]\n");

     gets(string);

     if( (set_learning_variables(string))==1  ){

	   printf("TEST SET STORED\n");

					       }

	   printf("PLEASE WAIT LOADING...\n");




  input_layer = first_layer_connections_taker( (set->HID),(set->INP) );

  hidden_layer = second_layer_connections_taker( (set->HID),(set->OUT) );

  upper = biggest_range_taker( (set->INP) );

  lower = smallest_range_taker( (set->INP) );



	 for(j=1;j<((set->HID)+1);j++){

		  k6=( ((set->INP)+1)*(j-1) );

	*(input_layer+k6)=-(*(input_layer+k6) );

				      }


	  for(j=1;j<((set->OUT)+1);j++){

		  k6=( ((set->HID)+1)*(j-1) );

	*(hidden_layer+k6)=-(*(hidden_layer+k6) );


				       }





	 printf("READY TO CLASSIFY press enter  \n");
	 printf("\n");
	 while(!kbhit());
	 getch();

	stream=fopen("RESPONSE.txt","w+");

	fprintf(stream,"%s  \n",goulp);



     t=0;
	     do{




	 name=take_a_filename(string,t);

	 strcpy(filename,name);


	 pattern = pattern_normaliser(filename,lower,upper,(set->INP) );




 hidden_activator = output_nodes_producer(input_layer,pattern,(set->HID),
					  (set->INP));

 output_activator = output_nodes_producer(hidden_layer,hidden_activator,
					  (set->OUT),(set->HID));




	 for(j=1;j<((set->OUT)+1);j++){



  if(j==1){

	  printf("%s=%f\n",classNames[j-1],(*(output_activator+j)) );
	  aigo="CHARATERIZED AS:";
	  fprintf(stream,"%s \n",aigo);
	  fprintf(stream,"%s %f \n",classNames[j-1],(*(output_activator+j)) );

	  }

  if(j==2){

	  printf("%s=%f\n",classNames[j-1],(*(output_activator+j)) );
	  fprintf(stream,"%s %f \n",classNames[j-1],(*(output_activator+j)) );

	  }


  if(j==3){

	  printf("%s=%f\n",classNames[j-1],(*(output_activator+j)) );
	  fprintf(stream,"%s %f \n",classNames[j-1],(*(output_activator+j)) );

	  }

   if(j==4){

	  printf("%s=%f\n",classNames[j-1],(*(output_activator+j)) );
	  fprintf(stream,"%s %f \n",classNames[j-1],(*(output_activator+j)) );

	   }




				 }

	 printf("\n");
	 printf("SOURCE FILE: %s\n",filename);
	 lena="source file:";
	 fprintf(stream,"%s %s \n",lena,filename);


	 printf("\n");
	 puts("PRESS ENTER to CLASSIFY a NEW INPUT\n");
	 while(!kbhit());
	 getch();


	  farfree(pattern);
	  farfree(hidden_activator);
	  farfree(output_activator);


	 t++;


	} while( t< (set->T) );

     fclose(stream);
     printf("\n");
     printf("CLASSIFICATIONS CORRECTLY SAVED ON FILE \n");
     puts("hit any key to quit");
     while(!kbhit());
     getch();

     farfree(input_layer);
     farfree(hidden_layer);
     farfree(upper);
     farfree(lower);



				    }// END


    int    set_learning_variables(char path_store[80])

  {


     FILE * sc;


     char ch,first_file[80];
     int k1=0,done;
     float cs;
     float far * dim;
     unsigned int  k2=0;   //unsigned  int//
     int s;
     struct find_t ffblk;


  if( (done = _dos_findfirst(path_store,_A_NORMAL,&ffblk) )==0 ){

  k2++;
  printf("FIRST  FILE of the list:%s\n",ffblk.name);

							       }


  else{

  printf("cannot find FIRST FILE of the list\n");
  exit(1);

       }


	   strxfrm(first_file,path_store,(strlen(path_store)-2));

	   strcat(first_file,ffblk.name);



  if((sc=fopen(first_file,"r"))==NULL){
  printf("cannot open first file of the list\n");
  exit(1);
				      }



  while(!feof(sc)){
  k1++;
  fscanf(sc,"%f\n",&cs);

		  }


  printf("INPUT NEURONS =%d\n",k1);


  set->INP=k1;

  while (!done)  {


     done = _dos_findnext(&ffblk);
     if(done==0){
     k2++;

		}

		  }


  printf("NUMBER OF PATTERNS TO BE TESTED =%d\n",k2);

  set->T=k2;

  printf("INSERT number of classes to discriminate (OUTPUT NEURONS)\n");
  scanf("%d",&k1);

  set->OUT=k1;

  for(s=0; s<k1; s++)
  {

  printf("Please Insert Name of the Class to Indentify\n");
  scanf("%s", &classNames[s]);
  printf("Class to Identify: %s\n", classNames[s]);

  }

  printf("INSERT NUMBER of HIDDEN NEURONS\n");
  scanf("%d",&k1);
  set->HID=k1;

  printf("CHECK if the TEST set is currently correct\n");

  printf("\n");

  printf("TEST PATTERNS=%d\n",set->T);
  printf("INP=%d\n",set->INP);
  printf("HID=%d\n",set->HID);
  printf("OUT=%d\n",set->OUT);

  printf("\n");

  printf("DIGIT Y to SET THE NETWORK or N to quit\n");
  ch=getche();
  ch=toupper(ch);
  printf("\n");

  switch(ch){

  case 'N':
  exit(0);

  case 'Y':
  break;

	    }

  return 1;

 }


  float huge *   first_layer_connections_taker( int p, int q )

  {
	float huge * first_layer;
	int j , k1, k6;
	FILE * sl;
	float boot;



if ((first_layer = (float huge *) farcalloc( ((p+1)*(q+1)),sizeof(float)) )
								      == NULL)
{
   printf("Not enough memory to allocate buffer first_layer\n");
   exit(1);
}



	if ((sl = fopen("WEIGHT1.txt","r"))==NULL)

	{

	printf("WARNING CANNOT OPEN FIRST LAYER CONNECTIONS FILE\n");
	exit(1);

	}



	    for(j=1;j<p+1;j++){

	      for(k1=0;k1<q+1;k1++){

		k6=(k1 +((q+1)*(j-1)) );

		fscanf(sl,"%f\n",&boot);

		*(first_layer + k6)=boot;

				     }

			      }
	   fclose(sl);

    return first_layer;

  }


    float huge *  second_layer_connections_taker( int r, int v )

  {

	 float huge * second_layer;
	 int j , k1, k6;
	 FILE * sl;
	 float boot;



 if ((second_layer = (float huge *) farcalloc( ((r+1)*(v+1)),sizeof(float)) )
								      == NULL)
{
   printf("Not enough memory to allocate buffer second_layer\n");
   exit(1);
}
	if ((sl = fopen("WEIGHT2.txt","r"))==NULL)

	{

	printf("WARNING CANNOT OPEN SECOND LAYER CONNECTIONS FILE\n");
	exit(1);


	}



	 for(k1=0;k1<r+1;k1++){

		 for(j=1;j<v+1;j++){


		k6=( (j-1)+(k1*v) );

		fscanf(sl,"%f\n",&boot);

		*(second_layer + k6)=boot;



				   }


			      }
	   fclose(sl);

    return second_layer;

  }


    float far * biggest_range_taker( int r )

  {
	float far * biggest;
	int j ;
	FILE * sl;
	float boot;


 if ((biggest = (float far *) farcalloc((r+1),sizeof(float)) )
							      == NULL)
{
   printf("Not enough memory to allocate buffer biggest\n");
   exit(1);
}
	if ((sl = fopen("BIGGEST.txt","r"))==NULL)

	{

	printf("WARNING CANNOT OPEN FIRST NORMALISER FILE\n");
	exit(1);

	}


	  for(j=1;j<r+1;j++){


		fscanf(sl,"%f\n",&boot);

		*(biggest + j)=boot;

			    }
	   fclose(sl);

    return biggest;

  }


   float far * smallest_range_taker( int r )

  {
	float far * smallest;
	int j ;
	FILE * sl;
	float boot;

 if ((smallest = (float far *) farcalloc((r+1),sizeof(float)) )
							      == NULL)
{
   printf("Not enough memory to allocate buffer smallest\n");
   exit(1);
}

	if ((sl = fopen("SMALLEST.txt","r"))==NULL)

	{

	printf("WARNING CANNOT OPEN SECOND NORMALISER FILE\n");
	exit(1);


	}


	  for(j=1;j<r+1;j++){


		fscanf(sl,"%f\n",&boot);

		*(smallest + j)=boot;

			    }
	   fclose(sl);


    return smallest;

  }

  float far * pattern_normaliser(char filename[128], float far *p1 ,float far *
		p2 , int x)
   {
     FILE *pf;
     int k2;
     float boot,number;
     float  far *pattern_normalisation;




   if ((pattern_normalisation = (float far *) farcalloc( (x+1),
					     sizeof(float)) ) == NULL)
{
   printf("Not enough memory to allocate buffer pattern\n");
   exit(1);
}



	      if ((pf = fopen(filename,"r"))==NULL)
	{
	 printf("Cannot open file %s\n",filename);
	 farfree(pattern_normalisation);
	 exit(1);

	}




	    for(k2=1;k2<x+1;k2++){

	      fscanf(pf,"%f\n",&number);

	      boot = normalise( number,*(p1+k2),*(p2+k2) );

	      if(boot<0.01){

    *(pattern_normalisation+k2)=0.0;

			    }

	      if(boot>0.99){

    *(pattern_normalisation+k2)=1.0;

			    }

	  if( (boot>=0.01) && (boot<=0.99) ){

    *(pattern_normalisation+k2)=boot;

					    }



				 }
   fclose (pf);

   return pattern_normalisation;

  }

	float     normalise(float value, float min, float max)

       {

       return (value-min)/(max-min);

       }


  char * take_a_filename( char current_dir[80],unsigned int p)

   {

     char klf[80];
     char * found;
     unsigned int x;
     struct find_t ffblk;


	strxfrm(klf,current_dir,(strlen(current_dir)-2));


	_dos_findfirst(current_dir,_A_NORMAL,&ffblk);

	  for(x=1;x<=p;x++){

			_dos_findnext(&ffblk);

			       }


	strcat(klf,ffblk.name);

	found=klf;


    return  found;

  }

   float sigmoid(float input)
    {

    float activate;
    activate=(float)(1/(1+exp((double)-input)));
    return activate;

    }


   float far * output_nodes_producer( float huge * l1 , float far * l2 ,
				      int p ,int r)
    {

    float far * y ;
    float top,THR;
    int i,j,k6,k5;

   if ((y = (float far *) farcalloc( (p+1), sizeof(float)) ) == NULL)
{
   printf("Not enough memory to allocate buffer OUTPUT NODES\n");
   exit(1);

}




	   for(j=1;j<p+1;j++){

			       top=0;

	   for(i=1;i<r+1;i++){


			       k6=(i+((r+1)*(j-1)));

	   top  = top + ( (*(l1+k6) ) * ( *(l2+i) ) );

			      }

			       k5=((r+1)*(j-1));

	   THR =( top - (*(l1+k5)) );

	   *(y+j) = sigmoid(THR);


			     }
      return y;

   }