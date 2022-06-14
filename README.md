# ProjektSIDwa
Trenowanie modelu
Utworzony został pik csv X_train.csv przechowujący dane z plików .xml
W funkcji BOW odbywa się klasteryzacja oraz zapis danych do słownika. 
Następnie w funkcji extract_train tworzone są deskryptora dla każdego wycinka, które zapisywane są w osobnej zmiennej. 
Ostateczne trenowanie modelu obdywa się w funkcji train() za pomocą drzewa losowego o rozmiarze 100.

W funkcji create_input_csv() tworzony jest plik csv X_test.csv, który zawiera dane wejściowe wpisane przez użytkownika. 
W funkcji extract_input() odbywa się ekstrakcja wycinków, tak jak w przypadku danych treningowych.
Funkcja predict_im() odpowiada za predykcję, co znajduje się w wycinku.
Na końcu wyświetlany jest napis "speedlimit" lub "other" w zależności od wyniku predykcji. 
