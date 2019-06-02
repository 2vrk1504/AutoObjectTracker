#include<Servo.h>
#include<math.h>
int state =0;
Servo yaw, pitch;
int inchar;
float dyaw = -30;
int st = 10;
int current_yaw = 170, current_pitch = 170;

void move()
{
  if(current_yaw < 10)
  {
    current_pitch=20;
    dyaw=0;
    current_yaw += dyaw;
    signal_to_servo(current_yaw, current_pitch);
    dyaw=30;
  }
  
  current_yaw += dyaw;
  signal_to_servo(current_yaw, current_pitch);
}

void signal_to_servo(int x1,int y1)
{
  yaw.write(x1);
  pitch.write(y1);
}

void movetoframe()  
{
  if(inchar =='0')
    {current_yaw=170;
    current_pitch=170;}
  if(inchar =='1')
    {current_yaw=140;
    current_pitch=170;}
  if(inchar =='2')
    {current_yaw=110;
    current_pitch=170;}
  if(inchar =='3')
    {current_yaw=80;
    current_pitch=170;}
  if(inchar =='4')
    {current_yaw=50;
    current_pitch=170;}
  if(inchar =='5')
    {current_yaw=20;
    current_pitch=170;}
  if(inchar =='6')
    {current_yaw=0;
    current_pitch=170;}
  if(inchar =='7')
    {current_yaw=30;
    current_pitch=0;}
  if(inchar =='8')
    {current_yaw=60;
    current_pitch=0;}
  if(inchar =='9')
    {current_yaw=90;
    current_pitch=0;}
  if(inchar =='a')
    {current_yaw=120;
    current_pitch=0;}
  if(inchar =='b')
    {current_yaw=150;
    current_pitch=0;}
  signal_to_servo(current_yaw,current_pitch);
  
}

void moveinframe()
{
   if(inchar =='1')
    {current_yaw+=st;
    current_pitch+=st;}
  if(inchar =='2')
    {current_yaw+=0;
    current_pitch+=st;}
  if(inchar =='3')
    {current_yaw += -st;
    current_pitch+=st;}
  if(inchar =='4')
    {current_yaw+=st;
    current_pitch+=0;}
  if(inchar =='5')
    {current_yaw+=0;
    current_pitch+=0;}
  if(inchar =='6')
    {current_yaw += -st;
    current_pitch+=0;}
  if(inchar =='7')
    {current_yaw+=st;
    current_pitch += -st;}
  if(inchar =='8')
    {current_yaw+=0;
    current_pitch += -st;}
  if(inchar =='9')
    {current_yaw+= -st;
    current_pitch += -st;}
    signal_to_servo(current_yaw,current_pitch);
}

void setup() {
  // put your setup code here, to run once:
  yaw.attach(9);
  state=0;
  pitch.attach(6);
   while (!Serial) 
  {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  //pinMode(6, INPUT);
  Serial.begin(57600);
  yaw.write(170);
  pitch.write(170);
  delay(100);
}

void loop() 
{
  //put your main code here, to run repeatedly
    inchar = Serial.read();
  //Serial.println(inchar);
    if(inchar == 'g' && !state)
    {
    move();
    } 
    else if((inchar =='0'||inchar =='1'||inchar =='2'||inchar =='3'||inchar =='4'||inchar =='5'||inchar =='6'||inchar =='7'||inchar =='8'||inchar =='9'||inchar =='a'||inchar =='b')&&!state)
       { movetoframe();
         state=1;}
    else if((inchar =='0'||inchar =='1'||inchar =='2'||inchar =='3'||inchar =='4'||inchar =='5'||inchar =='6'||inchar =='7'||inchar =='8'||inchar =='9')&&state){
              moveinframe();
              state=0;
         }
}
