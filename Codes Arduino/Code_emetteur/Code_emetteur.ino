// Pins declaration and global variables
const int speaker = 12;

// Carrier and bit rate frequencies (Hz)
const int carrier = 500;
const int bit_rate = 125;
const int period = 1000 / carrier; // Period (ms)

bool checkMessage(String string_to_test){
  /* This function allows to check if the string contain only 0 or 1 char.
   * It takes as inuput the string and return if yes or not the string is
   * valid.
   */
  const int string_len = string_to_test.length();
  for (int i = 0; i < string_len; i++){
    if (string_to_test.charAt(i) != '0' && 
        string_to_test.charAt(i) != '1'){
      return false;
    }
  }
  return true;
}

void setup() {
  pinMode(speaker, OUTPUT);
  digitalWrite(speaker, LOW);
  // Display frequency
  Serial.begin(9600);
}

void loop() {
  // Retrieving word to send
  Serial.println("Please, enter binary message:");
  /*
  // Waiting for message
  bool string_complete = false;
  String input_string = "";
  while(!string_complete){

    if (Serial.available()){
      
      const char c = Serial.read();

      if (c != '0' && c != '1'){
        string_complete = true;
      }else{
        input_string += c;
      }
    }
    delay(10);
  }
  */
  delay(1000);
  String input_string = "010101110001010111000101011100010101110001010111000101011100010101110001010111000101011100010101110001010111000101011100010101110001010111000101011100010101110001010111000101011100010101110001010111000101011100010101110001010111000101011100010101110001010111000101011100010101110001010111000101011100010101110001010111000101011100010101110001010111000101011100";
  // Comfirming it to user
  Serial.print("Recieved chain: ");
  delay(100);
  Serial.println(input_string);

  // Checking string
  if(!checkMessage(input_string)){
    Serial.println("Error: invalid string");
    delay(100);
    exit(-1);
  }

  // Sending signal using BPSK
  // We have HIGH LOW for 0 and LOW HIGH for 1
  const int string_len = input_string.length();
  
  // Computting number of cycle for the bit rate
  const int nb_cycle = carrier / bit_rate;

  // First loop is for the whole message and the second one to repeat
  // the bit to respect the bit rate
  for (int i = 0; i < string_len; i++){
    
    const char c = input_string.charAt(i); 
    for (int j = 0; j < nb_cycle; j++){
      
      // Making signal
      if (c == '0'){
        digitalWrite(speaker, HIGH);
        delay(period / 2);
        digitalWrite(speaker, LOW);
        delay(period / 2);
      }else{
        digitalWrite(speaker, LOW);
        delay(period / 2);
        digitalWrite(speaker, HIGH);
        delay(period / 2);
      }
    }
  }

  // Shutting down the speaker
  digitalWrite(speaker, LOW);

  Serial.println("Message sent successfully!");
  delay(100);
  exit(0);
}
