import sounddevice as sd
import numpy as np

def main():
    print("Ricerca dei dispositivi di input audio...")
    devices = sd.query_devices()
    input_devices = []
    
    print("\n--- MICROFONI DISPONIBILI ---")
    for i, dev in enumerate(devices):
        # Filtra solo i dispositivi che hanno canali di input
        if dev['max_input_channels'] > 0:
            # Mostra anche il Sample Rate di default del dispositivo
            fs = int(dev['default_samplerate'])
            print(f"[{i}] {dev['name']} (Nativo a {fs} Hz)")
            input_devices.append(i)
            
    if not input_devices:
        print("Nessun dispositivo di input trovato nel sistema!")
        return

    while True:
        choice = input("\nInserisci il numero del microfono da testare (o 'q' per uscire): ")
        if choice.lower() == 'q':
            print("Uscita.")
            break
            
        try:
            device_id = int(choice)
            if device_id not in input_devices:
                print("Numero non valido o dispositivo non di input. Riprova.")
                continue
        except ValueError:
            print("Per favore, inserisci un numero valido.")
            continue
            
        test_mic(device_id)

def test_mic(device_id, duration=3):
    # Recupera il Sample Rate corretto per QUESTO specifico dispositivo per evitare l'errore -9997
    device_info = sd.query_devices(device_id)
    fs = int(device_info['default_samplerate'])
    
    print(f"\n🎙️  Test del dispositivo [{device_id}] a {fs} Hz per {duration} secondi...")
    print("Parla ora! (Es: 'Uno, due, tre, prova')")
    
    try:
        # Avvia la registrazione con la frequenza nativa
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32', device=device_id)
        sd.wait()
        
        # Calcola la potenza del segnale
        rms = np.sqrt(np.mean(recording**2))
        peak = np.max(np.abs(recording))
        
        print(f"📊 Risultati -> Volume (RMS): {rms:.4f} | Picco massimo: {peak:.4f}")
        
        if rms > 0.005:
            print("✅ FUNZIONA! Questo microfono cattura la tua voce chiaramente.")
        elif rms > 0.0001:
            print("⚠️ Segnale molto debole. Potrebbe essere il microfono sbagliato o volume bassissimo.")
        else:
            print("❌ Silenzio assoluto. Non è il microfono giusto (o è scollegato).")
            
    except Exception as e:
        print(f"❌ Impossibile aprire questo dispositivo. Errore: {e}")

if __name__ == "__main__":
    main()