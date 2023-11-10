#define pinDeSortieArduino 12   // On va utiliser la broche D12 de l'Arduino Uno
#define frequenceDeDebut 200    // Fréquence "basse" de la sirène
#define frequenceDeFin 1700     // Fréquence "haute" de la sirène

void setup()
{
  // Définition de la broche D12 en "sortie"
  pinMode(pinDeSortieArduino, OUTPUT);
} 

void loop()
{
  /*
  // Phase de "montée" sirène
  for (int i = frequenceDeDebut; i < frequenceDeFin; i=i+3) {
    tone(pinDeSortieArduino, i); 
    delay(1); 
  }

  // Phase de "descente" sirène
  for (int i = frequenceDeFin; i > frequenceDeDebut; i=i-3) {
    tone(pinDeSortieArduino, i); 
    delay(1); 
  }
  */
  digitalWrite(pinDeSortieArduino, HIGH);
  delay(1000/100);
  digitalWrite(pinDeSortieArduino, LOW);
  delay(1000/100);
}