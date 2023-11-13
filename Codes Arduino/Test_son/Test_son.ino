#define pinDeSortieArduino 12   // On va utiliser la broche D12 de l'Arduino Uno
#define frequenceDeDebut 200    // Fréquence "basse" de la sirène
#define frequenceDeFin 1000     // Fréquence "haute" de la sirène

void setup()
{
  // Définition de la broche D12 en "sortie"
  pinMode(pinDeSortieArduino, OUTPUT);
} 

void loop()
{
  
  // Phase de "montée" sirène
  int j = 0;
  for (int i = frequenceDeDebut; i < frequenceDeFin; i=i+3) {
    if (j % 70 < 35){tone(pinDeSortieArduino, i);}
    else{tone(pinDeSortieArduino, i + 100);}
    delay(20);
    j++;
  }
  j = 0;
  // Phase de "descente" sirène
  for (int i = frequenceDeFin; i > frequenceDeDebut; i=i-3) {
    if (j % 32 < 16){tone(pinDeSortieArduino, i);}
    else{tone(pinDeSortieArduino, i + 100);}
    delay(40);
    j++;
  }
  
}