// Pins declaration and global variables
const int micro = A0;

// Sampling rate (Hz)
const int sample_rate = 100;

void setup() {
  pinMode(micro, INPUT);
  // Display frequency
  Serial.begin(9600);
}

void loop() {
  // Retrieving microphone value (V) then display it
  const float micro_value = analogRead(micro) * (5.0 / 1023.0);
  Serial.print("Micro voltage: ");
  Serial.println(micro_value);
  
  // Waiting for next sample
  delay(1000 / sample_rate);
}
