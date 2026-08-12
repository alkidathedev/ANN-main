
       #include <stdio.h>
       #include <alloc.h>
       #include <string.h>
       #include <dos.h>
       #include <stdlib.h>
       #include <conio.h>
       #include <math.h>
       #include <ctype.h>

       #define   LOWEST       -500000      // -feature min value
       #define   HIGHEST       500000     //  feature max value

       #define   INPUT         100        //  max inputs allowed
					  //  buff. wekk. sery


	 struct learning_set

     {

	unsigned int  T ;   //unsigned  int//
	int INP ;
	int OUT ;
	float RATE ;
	float SMOOF ;
	float TOLL ;
	unsigned int ITE ;
	unsigned int CONV;

      };

      struct learning_set * set;

      char buff[INPUT][10];

      int HID;


     float far * pattern_handler(char filename[128],int x);

     float far * make_desidered_output(int q, float far * op);

     float     normalise(float value, float min, float max);

     float huge * first_layer_random_weigth_generator(int r,int v);

     float huge * second_layer_random_weigth_generator(int v,int q);

     float huge * first_layer_weigth_change(int r,int v);

     float huge * second_layer_weigth_change(int v,int q);

     float far * pattern_normaliser(char filename[128], float *p1 ,float *
				    p2 , int x);

     float multiplier(float m);

     float sigmoid(float input);

     float far * F_net (float far * l3 ,int q);

     int index(float far * l7, unsigned int c);

     float far * delta2 ( float far * l4 , float far * l5 ,  float far * l6 ,
				      int q ,int b);

     float huge * correction_allocator(int r , int q);

     float huge * second_layer_correction_calculator( float far * l7,
			      float far * l8, int r ,int q);

     float huge * first_layer_correction_calculator( float far * l12,
			     float far * l13, int q ,int p);


     float far * delta1 ( float far * l9 , float huge * l10 ,
			    float far * l11 , int q ,int b);

     float error_calculator(float far * a1 , float far * a2, int q ,int b);


     float huge * second_layer_weight_adjustment( float huge * b1,
		       float huge * b2, float r1, float s1 ,int p ,int q );

     float huge * first_layer_weight_adjustment( float huge * b4,
			float huge * b5, float r1, float s1, int p ,int r);


     int  first_layer_connections_saver(float huge * x1, int p, int q);

     int  second_layer_connections_saver( float huge * x2, int r, int v );

     char * take_a_filename(char selector[10], char current_dir[80],
				      unsigned int r, unsigned int p);

     float far * set_learning_variables(char path_store[80]);

     int    find_requested_files(char selector[10], char current_dir[80]);

     int range_saver(float * f1 , float * f2 , int r );

     float huge * new_second_layer_weight_adjustment(float huge * b1,
	       float huge * b2,float r1,float s1,int f,int p,int q );

     float huge * new_first_layer_weight_adjustment(float huge * b4,
	       float huge * b5, float r1, float s1 ,int f, int p,int r);

	 float far * output_nodes_producer( float huge * l1 , float far * l2 ,
				       int p ,int r);

	



      void main (void)

     {

      int jo,ju,acc,acs;  //T . OUT . INP  .HID
      int k1,k2,k3,k4,k6,alt,re_learn;
      float num,wekk[INPUT+1],sery[INPUT+1];
      float erro,top;
      unsigned int j,gt;
      unsigned int counter,count;
      int inde;
      float ru,su;  /// .RATE .SMOOF  //
      unsigned int select;
      char ch,string[80];
      char what_to_do;
      char * name;
      char filename[128],FILENAME[128];
      int i;



      float  far * pattern_pointing;
      float  far * output;


      float  huge * first_layer_connections;
      float  huge * second_layer_connections;


      float  huge * first_layer_change;
      float  huge * second_layer_change;


      float  far  * pattern_user;
      float  far  * pattern_detector;

      float far   * transfer_hidden ;
      float far   * transfer_output ;

      float far   * function_net1;
      float far   * function_net2;

      float far   * delta2_store;
      float far   * delta1_store;

      float huge  * corr1;
      float huge  * corr2;

      float huge  * transit1;
      float huge  * transit2;

      float huge  * def1;
      float huge  * def2;

      float huge * transit3;
      float huge * transit4;

      float far * class_dimension;

      float * smallest;
      float * biggest;

      float huge * inout;


      counter=0;
      erro=0;
      count=0;


     printf("\n");
     printf("INSERT CURRENT LEARN DIRECTORY [drive:\\directory\\*.*]\n");
     gets(string);
     class_dimension = set_learning_variables(string);


     printf("\n");
     printf("DIGIT Y to SET THE NETWORK or N to quit\n");
     ch=getche();
     ch=toupper(ch);
     printf("\n");

     switch(ch){

	  case 'N':
	  farfree(class_dimension);
	  exit(0);

	  case 'Y':
	  printf("\n");
	  printf("LEARNING SET STORED\n");
	  printf("\n");
	  printf("press enter to MAKE A RANGE FOR NORMALISATION\n");
	  while(!kbhit());
	  getch();

     }


	    output = make_desidered_output(set->OUT,class_dimension);

	    farfree(class_dimension);



	    for(j=0;j<(set->T);j++){


	    inde=index(output,j);

	    select=(int)(*(output+inde));

	    name=take_a_filename(buff[inde+1],string,select,j);


	    strcpy(filename,name);

	    acc=set->INP;

	    pattern_pointing = pattern_handler(filename,acc);

	    for(k1=0;k1<((set->INP)+1);k1++){

	    printf("IO[%d][%d]=%f\n",j,k1, *(pattern_pointing+k1) );
				 }


	    farfree( pattern_pointing );

			    }



	    for(k2=1;k2<((set->INP)+1);k2++){

		wekk[k2]=LOWEST;
		sery[k2]=HIGHEST;

				  }


	    for(j=0;j<(set->T);j++){

	    inde=index(output,j);

	    select=(int)(*(output+inde));

	    name=take_a_filename(buff[inde+1],string,select,j);

	    strcpy(filename,name);

	    acc=set->INP;

	    pattern_pointing = pattern_handler(filename,acc);

	    for(k2=1;k2<((set->INP)+1);k2++){

	    if( (*(pattern_pointing+k2) )    > (  wekk[k2] ) ){
	    wekk[k2] = *(pattern_pointing+k2) ;
							    }

	  if( ( *(pattern_pointing+k2) )     < ( sery[k2] ) ){
	  sery[k2] = *(pattern_pointing+k2) ;
							      }

				  }//1 for

	   farfree( pattern_pointing );

			  } //2 for
	   printf("\n");
	   printf("RANGE FOR NORMALISATION\n"); //x
          
	   printf("\n");

	   for(k2=1;k2<((set->INP)+1);k2++){

   printf("biggest[%d]=%f \t smallest[%d]=%f\n ",k2,wekk[k2],k2,sery[k2]);


					   }

	   printf("\n");
	   printf("press enter to NORMALISE ALL PATTERNS\n");
	   while(!kbhit());
	   getch();



   if (!(biggest = new float[(set->INP+1)*sizeof(float)]))
  {
   printf("Not enough memory to allocate biggest\n");
   exit(1);
  } else

   if (!(smallest = new float[(set->INP+1)*sizeof(float)]))
  {
   printf("Not enough memory to allocate smallest\n");
   exit(1);
  } else

	    for(k2=1;k2<((set->INP)+1);k2++){

	    *(biggest+k2)=wekk[k2];
	    *(smallest+k2)=sery[k2];

					  }




	      for(j=0;j<(set->T);j++){

	    inde=index(output,j);

	    select=(int)(*(output+inde));

	    name=take_a_filename(buff[inde+1],string,select,j);


	    strcpy(filename,name);


	      acc=set->INP;

    pattern_user = pattern_normaliser(filename,smallest,biggest,acc);

    for(k2=0;k2<((set->INP)+1);k2++){

    printf("IO[%d][%d]=%f\n",j,k2,*(pattern_user+k2) );

			   }
    printf("\n");
    farfree( pattern_user);
				}





       alt=0;
	      for(j=0;j<(set->T);j++){


	    inde=index(output,j);

	    select=(int)(*(output+inde));

	    name=take_a_filename(buff[inde+1],string,select,j);


	    strcpy(filename,name);

	    acc=set->INP;

       pattern_user = pattern_normaliser(filename,smallest,biggest,acc);

	      for(k2=1;k2<((set->INP)+1);k2++){

       if( ( *(pattern_user+k2) )>1){

       printf("WARNING IO[%d][%d]=%f\n",j,k2,*(pattern_user+k2) );
       printf("IS from file %s\n",filename);
       alt++;
				    } //if

				      }  //for k2

       farfree( pattern_user );

			      }//for j

       if(alt!=0){

       printf("press enter to quit\n");
       while(!kbhit());
       getch();
       exit(0);
		  } else printf("All inputs correctly normalised\n");

       printf("press enter to SEARCH FOR MISMATCHING CASES\n");
       while(!kbhit());
       getch();
       printf("please WAIT...\n");




       for(gt=0;gt<(set->T);gt++){


	    inde=index(output,gt);

	    select=(int)(*(output+inde));

	    name=take_a_filename(buff[inde+1],string,select,gt);

	      strcpy(filename,name);

	    acc=set->INP;

       pattern_user = pattern_normaliser(filename,smallest,biggest,acc);

       for(j=gt+1;j<(set->T);j++){

	    inde=index(output,j);

	    select=(int)(*(output+inde));

	    name=take_a_filename(buff[inde+1],string,select,j);


	    strcpy(FILENAME,name);


	    acc=set->INP;

       pattern_detector = pattern_normaliser(FILENAME,smallest,biggest,acc);


	    k2=-1;
	    alt=-1;

       do{

	    k2++;

       if( k2<((set->INP)+1) ){

       if( (*(pattern_user+k2) )== (*(pattern_detector+k2) ) ){

       alt=alt+1;
							       }  //if
		   }   //if

	 }while(k2<((set->INP)+1));

	if(alt==(set->INP)){


	printf("WARNING!! %s th is MISMATCHING  %s \n",filename,FILENAME);



	for(k2=0;k2<((set->INP)+1);k2++){

	printf("IO=%f   IO=%f\n",*(pattern_user+k2),
				 *(pattern_detector+k2)   );
				}
	printf("press enter to quit\n");
	while(!kbhit());
	getch();
	farfree( pattern_user );
	farfree( pattern_detector );

	if( farheapcheck() == _HEAPCORRUPT )
	printf( "Heap is corrupted.\n" );
	else
	printf( "Heap is OK.\n" );
	exit(0);
		    }  //if alt==inp

		    else   farfree( pattern_detector );

		       } //for j

	farfree( pattern_user );

			  } // for gt

	printf("No mismatching cases found\n");
	printf("press enter to START THE LEARNING PROCESS\n");
	while(!kbhit());
	getch();


	    acc=set->INP;
	    ju=set->OUT;
	    ru=set->RATE;
	    su=set->SMOOF;



     ///////////////One hidden layer network//////////////


	first_layer_connections =
	first_layer_random_weigth_generator(acc,HID);

	second_layer_connections =
	second_layer_random_weigth_generator(HID,ju);

	first_layer_change = first_layer_weigth_change(acc,HID);

	second_layer_change = second_layer_weigth_change(HID,ju);


	def1 = correction_allocator(HID,acc);

	def2 = correction_allocator(HID,ju);



	do{

		gt=0;
		erro=0.0;


	transit1 = correction_allocator(HID,acc);

	transit2 = correction_allocator(HID,ju);



	do{


	inde=index(output,gt);

	select=(int)(*(output+inde));

	name=take_a_filename(buff[inde+1],string,select,gt);

	strcpy(filename,name);

	pattern_user = pattern_normaliser(filename,smallest,biggest,acc);

	transfer_hidden = output_nodes_producer(first_layer_connections,
	pattern_user,HID, acc);

	*(transfer_hidden)= 1.0 ;

	function_net1 = F_net (transfer_hidden,HID);

	transfer_output =
	output_nodes_producer(second_layer_connections,
	transfer_hidden,ju, HID);


	function_net2 = F_net (transfer_output, ju);

	delta2_store = delta2(output,transfer_output,function_net2,ju,inde);

	corr2 = second_layer_correction_calculator(delta2_store,
		transfer_hidden,HID,ju);



		for(k4=0;k4<(HID+1);k4++){

		  for(k6=1;k6<((set->OUT)+1);k6++){

			   k1 = ((k4*(set->OUT))+k6);

      *(transit2 + k1) =( ( *(transit2 +k1) ) + ( *(corr2 + k1) ) );

						  }

						}
	     farfree(corr2);




	delta1_store = delta1(delta2_store,second_layer_connections,
			      function_net1,HID,ju);

	corr1 =  first_layer_correction_calculator(delta1_store,
		 pattern_user,HID,acc);

	     for(k4=1;k4<(HID+1);k4++){

		   for(k6=0;k6<((set->INP)+1);k6++){

		    k1 = ( ((k4-1)*((set->INP)+1)) + k6 );

	  *(transit1 + k1) =( ( *(transit1 +k1) ) + ( *(corr1 + k1) ) );

						    }

					     }
	      farfree(corr1);


       erro = erro + error_calculator(transfer_output,output,ju,inde);


	farfree (pattern_user);
	farfree (transfer_hidden);
	farfree (transfer_output);
	farfree (function_net1);
	farfree (function_net2);
	farfree (delta2_store);
	farfree (delta1_store);


	gt++;

	}while(gt<(set->T)) ;


	     counter++;

	    if(counter>(set->ITE))
		 {

	 printf("ERROR=%f\n",(erro*0.5));

	 counter=0;
	 count++;
		 }



	      for(k4=1;k4<(HID+1);k4++){

		   for(k1=0;k1<((set->INP)+1);k1++){


			 k6=(k1 +(((set->INP)+1)*(k4-1)) );

		*(def1 + k6) = *(first_layer_connections + k6);

						   }

					    }



	   for(k1=0;k1<(HID+1);k1++){

		 for(k4=1;k4<((set->OUT)+1);k4++){

		     k6=( (k4-1) + (k1*(set->OUT)) );

	      *(def2 + k6) = *(second_layer_connections + k6);

						 }

					   }

  if(HID==1){
       transit3 = second_layer_weight_adjustment(transit2,
		  second_layer_change,ru,su,HID,ju);
		}
  if(HID>1){
       transit3 = new_second_layer_weight_adjustment(transit2,
		  second_layer_change,ru,su,re_learn,HID,ju);
		}



		for(k1=0;k1<(HID+1);k1++){

			for(k6=1;k6<((set->OUT)+1);k6++){

			k4=( (k6-1) + (k1*(set->OUT)) );

		       *(second_layer_connections+k4) =
			( (*(def2+k4) ) + (*(transit3+k4) ) );

							}

						}
	  farfree(transit3);


      if(HID==1){
      transit4 =  first_layer_weight_adjustment(transit1,
		  first_layer_change,ru,su,HID,acc);
		   }
      if(HID>1){
      transit4 =  new_first_layer_weight_adjustment(transit1,
		  first_layer_change,ru,su,re_learn,HID,acc);
		}



	       for(k6=1;k6<(HID+1);k6++){

		       for(k1=0;k1<((set->INP)+1);k1++){


		k4=( k1+ (((set->INP)+1) * (k6-1) ) );

		*(first_layer_connections+k4) =
		 ( (*(def1+k4) ) + (*(transit4+k4) ) );

						       }

					       }

	  farfree(transit4);



	   for(k1=0;k1<(HID+1);k1++){

		 for(k3=1;k3<((set->OUT)+1);k3++){

			     k2=((k1*(set->OUT))+k3);

	      *(second_layer_change+k2) = *(transit2+k2);


						 }

					 }

	farfree(transit2);


	      for(k3=1;k3<(HID+1);k3++){

		       for(k1=0;k1<((set->INP)+1);k1++){

		 k6=(k1 +(((set->INP)+1)*(k3-1)) );

	     *(first_layer_change + k6) = *(transit1 + k6);

						       }

					      }
	   farfree(transit1);



	     if(  (count>=set->CONV)
		)
	     {
	     count=0;


	     if(//1

	HID < ( (int)(2*sqrt( (double) (set->INP)+(set->OUT) ) ) )

	       )//1
	      {  //if


     do{

	printf("\n");
	printf("YOU CAN INTERACT with the LEARNING PROCESS\n");
	printf("DIGIT Y to ADD a NEURON to the NETWORK\n");
	printf("DIGIT N to NOT CHANGE the NETWORK ARCHITECTURE\n");
	printf("DIGIT S to STOP The LEARNING PROCESS\n");

	what_to_do=getche();
	what_to_do=toupper(what_to_do);

	printf("\n");

       }while( (what_to_do!='Y')&&(what_to_do!='N')&&(what_to_do!='S') );


     switch(what_to_do){

	  case 'Y':
	  printf("\n");


	HID = HID+1;

	printf("A NEURON IS ADDED TO THE HIDDEN LAYER\n");
	printf("%d neurones are in the HIDDEN LAYER\n",HID);

	printf("How many previous neurones want you relearn?\n");
	
	printf("Please select between %d and %d neurones\n", 0, (HID-1) );

	scanf("%d",&k1);
	
	if ( k1 >= 0 && k1 <= (HID-1) ) re_learn = int(HID-k1);
	
	else
	{

		printf("The number of neurones you have selected to re-learn is not acceptable\n");

		farfree(second_layer_change);
		farfree(first_layer_change);
		farfree(def1);
		farfree(first_layer_connections);
		farfree(def2);
		farfree(second_layer_connections);

		delete biggest;
	    delete smallest;

		if( farheapcheck() == _HEAPCORRUPT )
	    printf( "Heap is corrupted.\n" );
	    else
	    printf( "Heap is OK.\n" );

		printf("Please press enter to Exit\n");
	    while(!kbhit());
	    getch();

	    exit(0);
		
	}


	 if ((inout =
 (float huge *) farcalloc( ((acc+1)*(HID+2)),sizeof(float)) )
							      == NULL)
{
   printf("Not enough memory to allocate buffer first_layer\n");

	if( farheapcheck() == _HEAPCORRUPT ) {
	 printf( "Heap is corrupted.\n" );
					     }
	 else {
	 printf( "Heap is OK.\n" );
	      }

   exit(1);
}                  for(j=1;j<HID;j++){

		   for(i=0;i<acc+1;i++){

		   k6=( i+ ((acc+1)*(j-1)) );

		*(inout + k6)=
		*(first_layer_connections + k6);
				     }
				     }
		       farfree(first_layer_connections);


 if ((first_layer_connections =
 (float huge *) farcalloc( ((acc+1)*(HID+2)),sizeof(float)) )
							      == NULL)
{
   printf("Not enough memory to allocate buffer first_layer\n");

	if( farheapcheck() == _HEAPCORRUPT )  {
	 printf( "Heap is corrupted.\n" );
					      }
	 else{
	 printf( "Heap is OK.\n" );
	     }

   exit(1);
}                  for(j=1;j<HID;j++){

		   for(i=0;i<acc+1;i++){

		   k6=( i+ ((acc+1)*(j-1)) );

		*(first_layer_connections + k6)=
		*(inout + k6);
				     }
				     }

		   for(j=re_learn;j<=HID;j++){

		       for(i=0;i<acc+1;i++){

		   k6=( i+ ((acc+1)*(j-1)) );

		   *(first_layer_connections + k6)= 0;

				     }
						}

		  for(j=1;j<HID;j++){

		   for(i=0;i<acc+1;i++){

		   k6=( i+ ((acc+1)*(j-1)) );

		 *(inout + k6) = *(first_layer_change + k6);

				       }
				       }

				farfree(first_layer_change);


 if ((first_layer_change = (float huge *) farcalloc( ((acc+1)*(HID+2)),
						    sizeof(float)) ) == NULL)
{
   printf("Not enough memory to allocate buffer first_layer_change\n");

	if( farheapcheck() == _HEAPCORRUPT ) {
	 printf( "Heap is corrupted.\n" );
					     }
	 else{
	 printf( "Heap is OK.\n" );
	     }

   exit(1);
}                  for(j=1;j<HID;j++){

		   for(i=0;i<acc+1;i++){

		   k6=( i+ ((acc+1)*(j-1)) );

		   *(first_layer_change + k6)=
		   *(inout + k6);

				       }
				       }

		for(j=re_learn;j<=HID;j++){

		  for(i=0;i<acc+1;i++){

		  k6=( i+ ((acc+1)*(j-1)) );

		  *(first_layer_change+k6)=0.0;

				       }
					     }


		  for(j=1;j<HID;j++){

		   for(i=0;i<acc+1;i++){

		   k6=( i+ ((acc+1)*(j-1)) );

		*(inout + k6)=
		*(def1 + k6);
				     }
				     }
			  farfree(def1);


  if ((def1 = (float huge *) farcalloc((acc+1)*(HID+2), sizeof(float)) )
			     == NULL)
   {
   printf("Not enough memory to allocate buffer corrections\n");

	if( farheapcheck() == _HEAPCORRUPT ){
	 printf( "Heap is corrupted.\n" );
					    }
	 else{
	 printf( "Heap is OK.\n" );
	     }
   exit(1);
   }
		    for(j=1;j<HID;j++){

		   for(i=0;i<acc+1;i++){

		   k6=( i+ ((acc+1)*(j-1)) );

		*(def1+k6)=
		*(inout + k6);
				     }
				     }


		 for(j=re_learn;j<=HID;j++){

		   for(i=0;i<acc+1;i++){

		   k6=( i+ ((acc+1)*(j-1)) );

		  *(def1+k6)= 0.0;
				     }
					       }
		     farfree(inout);

      if ((inout =
 (float huge *) farcalloc( ((ju+1)*(HID+2)),sizeof(float)) )
							      == NULL)
{
   printf("Not enough memory to allocate buffer second_layer\n");


	if( farheapcheck() == _HEAPCORRUPT ) {
	 printf( "Heap is corrupted.\n" );
					     }
	 else {
	 printf( "Heap is OK.\n" );
	      }

   exit(1);
}

		     for(i=0;i<HID;i++){

		   for(j=1;j<ju+1;j++){

		   k6=( (j-1) + (i*ju) );

		   *(inout + k6)=
		   *(second_layer_connections + k6);

				       }
				       }
			  farfree(second_layer_connections);


 if ((second_layer_connections =
 (float huge *) farcalloc( ((ju+1)*(HID+2)),sizeof(float)) )
							      == NULL)
{
   printf("Not enough memory to allocate buffer second_layer\n");


	if( farheapcheck() == _HEAPCORRUPT ) {
	 printf( "Heap is corrupted.\n" );
					     }
	 else{
	 printf( "Heap is OK.\n" );
	     }

   exit(1);
}

		     for(i=0;i<HID;i++){

		   for(j=1;j<ju+1;j++){

		   k6=( (j-1) + (i*ju) );

		   *(second_layer_connections + k6)=
		   *(inout + k6);

				       }
				       }

		  for(i=re_learn;i<=HID;i++){

		   for(j=1;j<ju+1;j++){

		       k6=( (j-1) + (i*ju) );

		   *(second_layer_connections + k6)=0;

				    }
						}


	 for(i=0;i<HID;i++){

		   for(j=1;j<ju+1;j++){

		   k6=( j + (i*ju) );

		   *(inout + k6)=*(second_layer_change + k6);

				       }
				       }
				  farfree(second_layer_change);



 if ((second_layer_change = (float huge *) farcalloc( ((ju+1)*(HID+2)),
					     sizeof(float)) ) == NULL)
{
   printf("Not enough memory to allocate buffer second_layer_change \n");

	if( farheapcheck() == _HEAPCORRUPT ){
	 printf( "Heap is corrupted.\n" );
					    }
	 else{
	 printf( "Heap is OK.\n" );
	     }

   exit(1);
}                   for(i=0;i<HID;i++){

		   for(j=1;j<ju+1;j++){

		   k6=( j + (i*ju) );

		   *(second_layer_change + k6)=
		   *(inout + k6);

				       }
				       }

	      for(i=re_learn;i<=HID;i++){

		 for(j=1;j<ju+1;j++){

		     k6=( j + (i*ju) );

		     *(second_layer_change+k6)=0.0;

				       }
						 }


		     for(i=0;i<HID;i++){

		   for(j=1;j<ju+1;j++){

		   k6=( (j-1) + (i*ju) );

		*(inout + k6)=
		*(def2 + k6);
				     }
				     }
		     farfree(def2);

  if ((def2 = (float huge *) farcalloc((ju+1)*(HID+2), sizeof(float)) )
				== NULL)
   {
   printf("Not enough memory to allocate buffer corrections\n");

	if( farheapcheck() == _HEAPCORRUPT ){
	 printf( "Heap is corrupted.\n" );
					    }
	 else{
	 printf( "Heap is OK.\n" );
	     }
   exit(1);
   }
		for(i=0;i<HID;i++){

		   for(j=1;j<ju+1;j++){

		   k6=( (j-1) + (i*ju) );

		*(def2 + k6)=
		*(inout + k6);
				     }
				     }


		   for(i=re_learn;i<=HID;i++){

		   for(j=1;j<ju+1;j++){

		     k6=( (j-1) + (i*ju) );

		  *(def2 + k6)= 0.0;
				     }
					     }

		       farfree(inout);

	  case 'N':
	  continue;

	  case 'S':
	  continue;


     }  //switch

	   }  //if  <


	     if(//1

	HID >=( (int)(2*sqrt( (double) (set->INP)+(set->OUT) ) ) )

	       )//1
	      {  //if


     do{

     printf("\n");
     printf("YOU CAN INTERACT with the LEARNING PROCESS\n");
     printf("DIGIT Y to continue the learning process\n");
     printf("DIGIT S to STOP The LEARNING PROCESS\n");
     what_to_do=getche();
     what_to_do=toupper(what_to_do);
     printf("\n");

       }while( (what_to_do!='Y')&&(what_to_do!='S') );


     switch(what_to_do){

	  case 'Y':
	  continue;

	  case 'S':
	  continue;

		       }  //switch

	   }  //if >=

	 }  //if  count


	 }while(    (what_to_do!='S')

			   &&

		((erro*0.5)>=(set->TOLL))

		); //while


	printf("THE NETWORK REACHED THE REQUESTED MINIMUM\n");
	printf("THE HIDDEN LAYER HAS %d NEURONES\n",HID);
	printf("press enter to save RESULTS\n");
	while(!kbhit());
	getch();


	farfree( output );

	farfree( first_layer_connections );
	farfree( second_layer_connections);
	farfree( first_layer_change );
	farfree( second_layer_change);


    if( (first_layer_connections_saver(def1,HID,acc) ) == 1 ){

    printf("FIRST LAYER WEIGHT SAVED\n");

					 }  else    {

    printf(" WARNING FIRST LAYER WEIGHT NOT SAVED \n");

						     }

    if( (second_layer_connections_saver(def2,HID,ju) ) == 1 ){

    printf("SECOND LAYER WEIGHT SAVED\n");

					 }  else     {

    printf(" WARNING SECOND LAYER WEIGHT NOT SAVED \n");

						     }

    if( ( range_saver(biggest,smallest,acc) ) == 2 ){

    printf("RANGE FOR NORMALISATION SAVED\n");

					 }  else     {

    printf("WARNING RANGE FOR NORMALISATION NOT SAVED \n");

						      }

	printf("press enter to quit\n");
	while(!kbhit());
	getch();


	farfree(def1);
	farfree(def2);
	delete biggest;
	delete smallest;

	if( farheapcheck() == _HEAPCORRUPT ) {
	 printf( "Heap is corrupted.\n" );
					     }

     } //end of main//



   float far *  pattern_handler(char filename[128],int x)

     {

     FILE *pf;
     int k2;
     float number;
     float  far *pointer;




     if ((pointer = (float far *) farcalloc( (x+1),sizeof(float)) ) == NULL)
{
   printf("Not enough memory to allocate buffer pattern\n");

	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );

   exit(1);
}



	      if ((pf = fopen(filename,"r"))==NULL)
	{
	 printf("Cannot open file %s\n",filename);
	 farfree(pointer);
	 exit(1);

	}


	 *(pointer) = 1.0;

	 for(k2=1;k2<x+1;k2++){

	 fscanf(pf,"%f\n",&number);

	 *(pointer+k2)=number;

			      }
	 fclose (pf);

	 return pointer;

       }  //end of func


    float far * make_desidered_output(int q, float far * op)
  {

      float far * y_output;
      int k5,k1,as;
      int k2;



  if ((y_output = (float far *) farcalloc( (q*q)+(q+1),sizeof(float)) )
							       == NULL)
 {
   printf("Not enough memory to allocate buffer y_output\n");


	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );

   exit(1);
  }

	*(y_output)=*(op);


	 for(k5=1;k5<=q;k5++){

	 *(y_output+k5) = ( (*(op+k5)) + (*(y_output+k5-1)) );

			     }

	 k5=0;

	 for(k2=0;k2<q;k2++){

	 k5++;

	 for(k1=0;k1<q;k1++){


	 as=(q+1)+k1+(k2*q);

	 if( k1==(k5-1) ) {

	 *(y_output+as)=1.0;

				   } else  *(y_output+as)=0.0;

			     }

			     }


	return  y_output;
 }//end of func


       float     normalise(float value, float min, float max)
       {

       return (value-min)/(max-min);

       }


 float huge * first_layer_random_weigth_generator(int r,int v)

       {

	float huge * first_layer;
	float s;
	int k6,i,j;

 if ((first_layer = (float huge *) farcalloc( ((r+1)*(v+1)),sizeof(float)) )
								      == NULL)
{
   printf("Not enough memory to allocate buffer first_layer\n");

	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );


   exit(1);
}



	  randomize();
		   for(j=1;j<v+1;j++){
		   for(i=0;i<r+1;i++){

		   s=rand() %10;

		   k6=( i+ ((r+1)*(j-1)) );

		   *(first_layer+k6)=s*0.01;

				       }
				       }
       return  first_layer;

 }//end of funct



    float huge * second_layer_random_weigth_generator(int v,int q)

       {


	  float huge * second_layer;
	  float s;
	  int k6,i,j;

 if ((second_layer = (float huge *) farcalloc( ((q+1)*(v+1)),sizeof(float)) )
								      == NULL)
{
   printf("Not enough memory to allocate buffer second_layer\n");


	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );


   exit(1);
}



	  randomize();

		   for(i=0;i<v+1;i++){
		   for(j=1;j<q+1;j++){

		   s=rand() %10;

		   k6=( (j-1) + (i*q) );

		   *(second_layer+k6)=s*0.01;

				       }
				       }
       return  second_layer;

 }//end of funct



    float huge * first_layer_weigth_change(int r,int v)

       {


	 float huge * first_layer_change;
	 int k6,i,j;

 if ((first_layer_change = (float huge *) farcalloc( ((r+1)*(v+1)),
						    sizeof(float)) ) == NULL)
{
   printf("Not enough memory to allocate buffer first_layer_change\n");

	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );


   exit(1);
}



		   for(j=1;j<v+1;j++){
		   for(i=0;i<r+1;i++){

		   k6=( i+ ((r+1)*(j-1)) );

		   *(first_layer_change+k6)=0.0;

				       }
				       }
       return  first_layer_change;

 }//end of funct


    float huge * second_layer_weigth_change(int v,int q)

       {


	float huge * second_layer_change;
	int k6,i,j;

 if ((second_layer_change = (float huge *) farcalloc( ((q+1)*(v+1)),
						     sizeof(float)) ) == NULL)
{
   printf("Not enough memory to allocate buffer second_layer_change \n");

	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );


   exit(1);
}
		   for(i=0;i<v+1;i++){
		   for(j=1;j<q+1;j++){

		   k6=( j + (i*q) );

		   *(second_layer_change+k6)=0.0;

				       }
				       }
       return  second_layer_change;

 }//end of funct


  float far * pattern_normaliser(char filename[128], float *p1 ,float *
		p2 , int x)
   {
     FILE *pf;
     int k2;
     float number;
     float  far *pattern_normalisation;




   if ((pattern_normalisation = (float far *) farcalloc( (x+1),
					     sizeof(float)) ) == NULL)
{
   printf("Not enough memory to allocate buffer pattern\n");


	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );


   exit(1);
}





	      if ((pf = fopen(filename,"r"))==NULL)
	{
	 printf("Cannot open file %s\n",filename);
	 farfree(pattern_normalisation);
	 exit(1);

	}



	    *(pattern_normalisation) = 1.0;

	    for(k2=1;k2<x+1;k2++){

	      fscanf(pf,"%f\n",&number);
    *(pattern_normalisation+k2)=normalise( number,*(p1+k2),*(p2+k2) );
				  }
       fclose (pf);

	    return pattern_normalisation;

  }  //end of func


    float multiplier(float m)
    {

    return  ( m * (1-m) );

    }

    float sigmoid(float input)
    {

    float activate;


    activate = (float)( 1 / (1 + exp((double)-input)) );


    return activate;

    }


	float far * output_nodes_producer( float huge * l1 , float far * l2 ,
				      int p ,int r)
    {
    float far * y ;
    float top ;
    int i,j,k6;

   if ((y = (float far *) farcalloc( (p+1), sizeof(float)) ) == NULL)
{
   printf("Not enough memory to allocate buffer OUTPUT NODES\n");
   exit(1);

}
	   for(j=1;j<p+1;j++){
			       top=0;
	   for(i=0;i<r+1;i++){
			       k6=(i+((r+1)*(j-1)));
	   top  = top + ( (*(l1+k6) ) * ( *(l2+i) ) );

			      }
	   *(y+j) = sigmoid( top );


			      }
      return y;

   }


   float far * F_net (float far * l3 ,int q)
   {

    float far * f1;
    float net ;
    int j ;


   if ((f1 = (float far *) farcalloc( (q+1), sizeof(float)) ) == NULL)
{
   printf("Not enough memory to allocate buffer F_net\n");

	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );


   exit(1);
}

	 for(j=1;j<q+1;j++){

	 net= *(l3+j);
	 *(f1+j) = multiplier ( net );

			   }
    return f1;

  }

   int index(float far * l7, unsigned int c)

  {

  int i,k2;

   k2=0 ;

  for(i=0;((int)(*(l7+i)))<=c;i++){

  k2++;
				  }

   return k2-1;

   }

   float far * delta2 ( float far * l4 , float far * l5 ,  float far * l6 ,
   int q ,int b)

   {

   float far *d;
   int a ,j ;

    if ((d = (float far *) farcalloc( (q+1), sizeof(float)) ) == NULL)
   {
   printf("Not enough memory to allocate buffer delta2\n");


	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );

   exit(1);
   }

	 for(j=1;j<q+1;j++){
			    a=(q+j+(q*b));

	*(d+j) = ( (*(l4+a)) - (*(l5+j)) ) * ( *(l6+j) );

			    }
	return d;

  }

	   float huge * correction_allocator(int r , int q)

     {

	   float huge * g;
	   int i;


  if ((g = (float huge *) farcalloc((r+1)*(q+1), sizeof(float)) ) == NULL)
   {
   printf("Not enough memory to allocate buffer corrections\n");

	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );

   exit(1);
   }


   for(i=0;i<(r+1)*(q+1);i++){
   *(g+i)=0.0;
			      }
   return g;

     }



 float huge * second_layer_correction_calculator( float far * l7, float far *
						  l8, int r ,int q)

 {
     float huge * h;
     int n,i,j;

    if ((h = (float huge *) farcalloc((r+1)*(q+1), sizeof(float)) ) == NULL)
   {
   printf("Not enough memory to allocate buffer corrections\n");


	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );


   exit(1);
   }

     for(i=0;i<r+1;i++){

	     for(j=1;j<q+1;j++){

     n = ((i*q)+j);
     *(h+n) =  ( (*(l7+j) ) * ( *(l8+i) ) );

				}

			}

     return h;

  }

float huge * first_layer_correction_calculator( float far * l12, float far *
						l13, int q ,int p)

 {
     float huge * k;
     int n,i,j;

     if ((k = (float huge *) farcalloc((p+1)*(q+1), sizeof(float)) ) == NULL)
   {
   printf("Not enough memory to allocate buffer corrections\n");


	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );


   exit(1);
   }


     for(j=1;j<q+1;j++){

	     for(i=0;i<p+1;i++){

	 n = ( ((j-1)*(p+1)) + i );
     *(k+n) =   ( (*(l12 +j) ) * ( *(l13 +i) ) );

				}

			}

     return k;

  }



  float far * delta1 ( float far * l9 , float huge * l10 ,  float far * l11 ,
		       int q ,int b)

   {

   float far * e;
   int i,j,n ;
   float sop;

    if ((e = (float far *) farcalloc( (q+1), sizeof(float)) ) == NULL)
   {
   printf("Not enough memory to allocate buffer delta1\n");

	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );

   exit(1);
   }

	 for(j=1;j<q+1;j++){

			    sop=0;

	 for(i=1;i<b+1;i++){

	 n=( i + (q*(i-1)) + j - 1 );
	 sop = sop + ( ( (*(l9 + i)) * (*(l10 + n)) ) );

			    }

	  *(e+j) = sop * ( *(l11 + j) ) ;

			     }
	return e;

  }

     float error_calculator(float far * a1 , float far * a2, int q ,int b)

     {

     int i,w;
     float fog;

     fog=0;

	 for(i=1;i<q+1;i++){

	 w=(q+i+(q*b));


	 fog = fog + (   ( ( *(a1 + i) ) - ( *(a2 + w) ) )  *
			 ( ( *(a1 + i) ) - ( *(a2 + w) ) )   );

			   }

     return fog;

     }

  float huge * second_layer_weight_adjustment( float huge * b1, float huge *
				      b2, float r1, float s1 , int p ,int q )

 {
     float huge * step1;

     int k1,k6,i,j;

       if ((step1 = (float huge *) farcalloc((p+1)*(q+1), sizeof(float)) ) == NULL)
   {
   printf("Not enough memory to allocate buffer second adjustments\n");


	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );

   exit(1);
   }

     for(i=0;i<p+1;i++){

	     for(j=1;j<q+1;j++){
				 k1=((i*q)+j);
				 k6= k1-1;

*(step1+k6) =  ( (r1 * (*(b1+k1) ) )  +   (s1 * (*(b2+k1) ) ) );

					}

			}

     return step1;

  }

   float huge * new_second_layer_weight_adjustment( float huge * b1,
		float huge * b2, float r1, float s1 ,int f, int p ,int q )

 {
     float huge * step1;

     int k1,k6,i,j;

       if ((step1 = (float huge *) farcalloc((p+1)*(q+1), sizeof(float)) ) == NULL)
   {
   printf("Not enough memory to allocate buffer second adjustments\n");


	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );

   exit(1);
   }


     for(i=0;i<p;i++){

	     for(j=1;j<q+1;j++){
				 k1=((i*q)+j);
				 k6= k1-1;

   *(step1+k6) =0.0;
					}

			}
			       for(i=f;i<=p;i++){

				 for(j=1;j<q+1;j++){

				 k1=((i*q)+j);
				 k6= k1-1;
	  *(step1+k6) =  ( (r1 * (*(b1+k1) ) )  +   (s1 * (*(b2+k1) ) ) );
						    }
						 }


     return step1;

  }


   float huge * first_layer_weight_adjustment( float huge * b4, float huge *
					b5, float r1, float s1 , int p ,int r)


 {
     float huge * step2;

     int k6,i,j;

 if ((step2 = (float huge *) farcalloc((r+1)*(p+1), sizeof(float)) ) == NULL)
   {
   printf("Not enough memory to allocate buffer first adjustments\n");


	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );

   exit(1);
   }


	     for(j=1;j<p+1;j++){

		 for(i=0;i<r+1;i++){

				    k6=(i + ((r+1)*(j-1)) );

	*(step2+k6) =  ( (r1 * (*(b4+k6) ) ) +

				   (s1 * (*(b5+k6) ) ) );

				    }


				}

     return step2;

  }


  float huge * new_first_layer_weight_adjustment( float huge * b4, float huge *
				b5, float r1, float s1 ,int f,int p ,int r)


 {
     float huge * step2;

     int k6,i,j;

 if ((step2 = (float huge *) farcalloc((r+1)*(p+1), sizeof(float)) ) == NULL)
   {
   printf("Not enough memory to allocate buffer first adjustments\n");


	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );

   exit(1);
   }
		  for(j=1;j<p;j++){

		    for(i=0;i<r+1;i++){

		     k6=(i + ((r+1)*(j-1)) );
			*(step2+k6) = 0.0;

			    }

				}

		  for(j=f;j<=p;j++){

		    for(i=0;i<r+1;i++){

		    k6=(i + ((r+1)*(j-1)) );

   *(step2+k6) =  ( (r1 * (*(b4+k6) ) ) +  (s1 * (*(b5+k6) ) ) );

				       }
				    }
	return step2;

  }


  int  first_layer_connections_saver( float huge * x1, int p, int q )

  {
	int j , k1, ok, k6;
	FILE * sl;


	if ((sl = fopen("WEIGHT1.txt","w+"))==NULL)

	{

	ok = 0 ;

	}  else  ok=1;

	    for(j=1;j<p+1;j++){

	      for(k1=0;k1<q+1;k1++){

		k6=(k1 +((q+1)*(j-1)) );

		fprintf(sl,"%f\n",*(x1 + k6));

				     }

			      }
	   fclose(sl);

    return ok;

  }

     int  second_layer_connections_saver( float huge * x2, int r, int v )

  {
	int j , k1, ok, k6;
	FILE * sl;


	if ((sl = fopen("WEIGHT2.txt","w+"))==NULL)

	{

	ok = 0 ;

	}  else  ok=1;


	 for(k1=0;k1<r+1;k1++){

		 for(j=1;j<v+1;j++){


		k6=( (j-1)+(k1*v) );

		fprintf(sl,"%f\n",*(x2 + k6));

				   }

			      }
	   fclose(sl);

    return ok;

  }

   char * take_a_filename(char selector[10], char current_dir[80],
					   unsigned int r,unsigned int p)

   {

     char klf[80];
     char * found;
     unsigned int x;
     struct find_t ffblk;
     char plk[80];

	strxfrm(klf,current_dir,(strlen(current_dir)-1));

	strcat(klf,selector);

	strxfrm(plk,current_dir,(strlen(current_dir)-2));

	_dos_findfirst(klf,_A_NORMAL,&ffblk);

	  for(x=1;x<=(p-r);x++){

			_dos_findnext(&ffblk);

			       }

	strcat(plk,ffblk.name);

	found=plk;


    return  found;

  }

  float far * set_learning_variables(char path_store[80])

  {


     FILE * sc;


     char first_file[80];
     int k1=0,l,done;
     float cs;
     float far * dim;
     unsigned int  k2=0;   //unsigned  int//

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


  printf("LEARNING PATTERNS =%d\n",k2);

  set->T=k2;

  printf("INSERT number of classes to discriminate (OUTPUT NEURONS)\n");
  scanf("%d",&k1);

  set->OUT=k1;

  if ((dim = (float far *) farcalloc( (k1+1), sizeof(float)) ) == NULL)
   {
   printf("Not enough memory to allocate buffer output limits\n");


	if( farheapcheck() == _HEAPCORRUPT )
	 printf( "Heap is corrupted.\n" );
	 else
	 printf( "Heap is OK.\n" );

   exit(1);
   }

   *(dim)=0;



   for(l=1;l<=k1;l++){

   printf("\n");

   printf("INSERT file extention of %d class [ . ,letter, number]\n",l);

   scanf("%s",&buff[l][0]);

   *(dim+l) = find_requested_files(buff[l],path_store);

   printf("\n");
   printf("%d FILES FOUND\n",(int)(*(dim+l)));
   printf("\n");

		     }


  HID=1;

  printf("INSERT LEARNING RATE [0...1]\n");
  scanf("%f",&cs);
  set->RATE=cs;

  printf("INSERT SMOOTHING FACTOR [0...1]\n");
  scanf("%f",&cs);
  set->SMOOF=cs;

  printf("INSERT LEARNING PROCESS TOLERANCE \n");
  scanf("%f",&cs);
  set->TOLL=cs;

  printf("INSERT ITERATION COUNTER VALUE (integer)\n");
  scanf("%d",&k1);
  set->ITE=k1;

  printf("INSERT LEARNING PROCESS CONVERGENCE CHECK VALUE (integer)\n");
  scanf("%d",&k1);
  set->CONV=k1;


  printf("CHECK if the learning set is currently correct\n");
  printf("\n");

  printf("T=%d\n",set->T);
  printf("INP=%d\n",set->INP);
  printf("HID=%d\n",HID);
  printf("OUT=%d\n",set->OUT);
  printf("RATE=%f\n",set->RATE);
  printf("SMOOF=%f\n",set->SMOOF);
  printf("TOLL=%f\n",set->TOLL);
  printf("ITE=%d\n",set->ITE);
  printf("CONV=%d\n",set->CONV);


  printf("\n");



   return dim ;


 }

     int    find_requested_files(char selector[10], char current_dir[80])

     {

     int f,done,g;
     char klf[80];
     unsigned int  k3=0;   //unsigned int//

     struct find_t ffblk;


	   strxfrm(klf,current_dir,(strlen(current_dir)-1));

	   strcat(klf,selector);

	   printf("\n\n");

     if((done = _dos_findfirst(klf,_A_NORMAL,&ffblk) )==0) {

     k3++;
     printf("%s\n",ffblk.name);

							   }

     while (!done)      {

     done = _dos_findnext(&ffblk) ;
     if(done==0){
     k3++;
     printf("%s\n",ffblk.name);
		}
			}


     return k3;

 }

     int range_saver(float * f1 , float * f2 , int r )

  {
	int j,ok,ko;
	FILE * sl;


	if ((sl = fopen("BIGGEST.txt","w+"))==NULL)

	{

	ok = 0 ;

	}  else  ok=1;


	  for(j=1;j<r+1;j++){


		fprintf(sl,"%f\n",*(f1 + j));

			    }
	   fclose(sl);

	if ((sl = fopen("SMALLEST.txt","w+"))==NULL)

	{

	ko = 0 ;

	}  else  ko=1;


	  for(j=1;j<r+1;j++){


		fprintf(sl,"%f\n",*(f2 + j));

			    }
	   fclose(sl);


    return (ok+ko);

  }