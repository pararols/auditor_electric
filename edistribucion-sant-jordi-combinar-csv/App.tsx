
import React, { useState, useCallback } from 'react';
import { Upload, FileText, X, Download, Combine, CheckCircle2, AlertCircle, Clock, Info, Layers } from 'lucide-react';
import { FileData, EnergyRecord, MergedRow } from './types';

// Parser ajustat per a format regional (separador ; i decimal ,)
const parseCSV = (text: string): EnergyRecord[] => {
  const lines = text.split(/\r?\n/).filter(line => line.trim() !== '');
  if (lines.length < 2) return [];

  const headers = lines[0].split(';').map(h => h.trim());
  const idxCUPS = headers.indexOf('CUPS');
  const idxFecha = headers.indexOf('Fecha');
  const idxHora = headers.indexOf('Hora');
  const idxAE = headers.indexOf('AE_kWh');
  const idxAuto = headers.indexOf('AE_AUTOCONS_kWh');

  if (idxFecha === -1 || idxHora === -1) return [];

  let records = lines.slice(1).map(line => {
    const values = line.split(';').map(v => v.trim());
    return {
      CUPS: idxCUPS !== -1 ? values[idxCUPS] : '',
      Fecha: values[idxFecha] || '',
      Hora: values[idxHora] || '',
      AE_kWh: idxAE !== -1 ? values[idxAE] : '0',
      AE_AUTOCONS_kWh: idxAuto !== -1 ? values[idxAuto] : '0'
    };
  });

  // NORMALITZACIÓ D'HORES (1-24 a 0-23)
  const hoursInt = records.map(r => parseInt(r.Hora)).filter(h => !isNaN(h));
  const hasZero = hoursInt.includes(0);
  const hasTwentyFour = hoursInt.includes(24);
  const minHour = hoursInt.length > 0 ? Math.min(...hoursInt) : 0;

  if (!hasZero && (hasTwentyFour || minHour === 1)) {
    records = records.map(r => {
      const hInt = parseInt(r.Hora);
      if (!isNaN(hInt)) {
        return { ...r, Hora: (hInt - 1).toString() };
      }
      return r;
    });
  }

  return records;
};

const parseDate = (d: string) => {
  const parts = d.split('/');
  if (parts.length !== 3) return new Date(0);
  return new Date(parseInt(parts[2]), parseInt(parts[1]) - 1, parseInt(parts[0]));
};

const App: React.FC = () => {
  const [files, setFiles] = useState<FileData[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = event.target.files;
    if (!selectedFiles) return;

    setError(null);
    const newFiles: FileData[] = [];

    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];
      try {
        const text = await file.text();
        const records = parseCSV(text);
        
        if (records.length === 0) {
          setError(`El fitxer ${file.name} no sembla vàlid.`);
          continue;
        }

        newFiles.push({
          id: Math.random().toString(36).substr(2, 9),
          name: file.name,
          records: records,
          cups: records[0]?.CUPS || 'N/A'
        });
      } catch (err) {
        setError(`Error en llegir el fitxer ${file.name}`);
      }
    }

    setFiles(prev => [...prev, ...newFiles]);
    event.target.value = '';
  };

  const removeFile = (id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id));
  };

  const generateCombinedCSV = useCallback(() => {
    if (files.length === 0) return;
    setIsProcessing(true);
    
    // Identifiquem els CUPS únics per crear les columnes de sortida
    // Fix: Explicitly type uniqueCups as string[] to avoid unknown index errors in subsequent loops
    const uniqueCups: string[] = Array.from(new Set(files.map(f => f.cups))).sort();
    
    const timeMap = new Map<string, MergedRow>();
    
    files.forEach(file => {
      const occurrenceTracker = new Map<string, number>();
      file.records.forEach(record => {
        const baseKey = `${record.Fecha}_${record.Hora}`;
        const instance = (occurrenceTracker.get(baseKey) || 0) + 1;
        occurrenceTracker.set(baseKey, instance);
        
        const fullKey = `${baseKey}_${instance}`;
        
        if (!timeMap.has(fullKey)) {
          timeMap.set(fullKey, {
            fecha: record.Fecha,
            hora: record.Hora,
            cupsValues: {}
          });
        }
        const row = timeMap.get(fullKey)!;
        
        // Si el CUPS ja té dades per aquesta hora, prioritzem valors no buits/zeros
        // Atès que l'usuari diu que no es solapen, una assignació directa funciona
        if (!row.cupsValues[file.cups] || (record.AE_kWh !== '0' && record.AE_kWh !== '')) {
            row.cupsValues[file.cups] = {
              ae: record.AE_kWh,
              autocons: record.AE_AUTOCONS_kWh
            };
        }
      });
    });

    const sortedKeys = Array.from(timeMap.keys()).sort((a, b) => {
      const rowA = timeMap.get(a)!;
      const rowB = timeMap.get(b)!;
      const dateA = parseDate(rowA.fecha).getTime();
      const dateB = parseDate(rowB.fecha).getTime();
      if (dateA !== dateB) return dateA - dateB;
      const hourA = parseInt(rowA.hora);
      const hourB = parseInt(rowB.hora);
      if (hourA !== hourB) return hourA - hourB;
      const instA = parseInt(a.split('_')[2]);
      const instB = parseInt(b.split('_')[2]);
      return instA - instB;
    });

    // Fila 1: CUPS (un cop per cada codi CUPS, sobre les dues columnes)
    let csv = `;;`; 
    uniqueCups.forEach(cups => {
      csv += `${cups};${cups};`;
    });
    csv += `\n`;

    // Fila 2: Capçaleres de dades
    csv += `Fecha;Hora`;
    uniqueCups.forEach(() => {
      csv += `;AE_kWh;AE_AUTOCONS_kWh`;
    });
    csv += `\n`;

    // Dades
    sortedKeys.forEach(key => {
      const row = timeMap.get(key)!;
      csv += `${row.fecha};${row.hora}`;
      
      uniqueCups.forEach(cups => {
        // Fix: Ensure 'cups' is treated as string for indexing row.cupsValues
        const values = row.cupsValues[cups as string] || { ae: '0', autocons: '0' };
        const ae = values.ae.toString().replace('.', ',');
        const auto = values.autocons.toString().replace('.', ',');
        csv += `;${ae};${auto}`;
      });
      csv += `\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `energia_combinada_cups.csv`);
    link.click();
    
    setIsProcessing(false);
  }, [files]);

  return (
    <div className="min-h-screen p-4 md:p-8 flex flex-col items-center">
      <header className="w-full max-w-4xl mb-8 text-center">
        <h1 className="text-3xl font-bold text-slate-800 flex items-center justify-center gap-3">
          <Combine className="w-8 h-8 text-blue-600" />
          Combinador per CUPS
        </h1>
        <p className="text-slate-500 mt-2">Dades d'autoconsum alineades i agrupades per punt de subministrament.</p>
      </header>

      <main className="w-full max-w-4xl space-y-6">
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <div className="flex flex-col items-center justify-center border-2 border-dashed border-slate-300 rounded-lg p-10 bg-slate-50 hover:bg-slate-100 relative cursor-pointer group">
            <input type="file" multiple accept=".csv" onChange={handleFileChange} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" />
            <Upload className="w-12 h-12 text-slate-400 group-hover:text-blue-500 mb-4 transition-colors" />
            <p className="text-lg font-medium text-slate-700">Puja els teus CSV</p>
            <p className="text-xs text-slate-400 mt-2">Els fitxers amb el mateix CUPS es fusionaran automàticament.</p>
          </div>

          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-2">
            <div className="flex items-center gap-2 text-[11px] text-blue-600 bg-blue-50 p-2 rounded border border-blue-100">
              <Layers className="w-4 h-4 flex-shrink-0" />
              <span>Agrupació intel·ligent: Fitxers del mateix CUPS ocupen un sol grup de columnes.</span>
            </div>
            <div className="flex items-center gap-2 text-[11px] text-amber-600 bg-amber-50 p-2 rounded border border-amber-100">
              <Clock className="w-4 h-4 flex-shrink-0" />
              <span>Normalització horària 0-23 aplicada per defecte.</span>
            </div>
          </div>
          
          {error && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm flex gap-2">
              <AlertCircle className="w-5 h-5 flex-shrink-0" /> {error}
            </div>
          )}
        </div>

        {files.length > 0 && (
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
            <div className="bg-slate-50 px-6 py-4 border-b border-slate-200 flex justify-between items-center">
              <h2 className="font-semibold text-slate-700 flex items-center gap-2">
                <FileText className="w-5 h-5 text-slate-500" /> {files.length} fitxers pujats
              </h2>
              <button onClick={() => setFiles([])} className="text-sm text-red-600 font-medium hover:underline">Netejar tot</button>
            </div>
            <ul className="divide-y divide-slate-100 max-h-[350px] overflow-y-auto">
              {files.map(file => (
                <li key={file.id} className="px-6 py-4 flex items-center justify-between hover:bg-slate-50">
                  <div className="overflow-hidden">
                    <p className="font-medium text-slate-800 truncate text-sm">{file.name}</p>
                    <p className="text-[10px] text-slate-400 font-mono bg-slate-100 px-1 rounded inline-block">CUPS: {file.cups}</p>
                  </div>
                  <button onClick={() => removeFile(file.id)} className="text-slate-300 hover:text-red-500 p-2 transition-colors"><X className="w-4 h-4" /></button>
                </li>
              ))}
            </ul>
            <div className="p-6 bg-slate-50 border-t border-slate-200">
              <button onClick={generateCombinedCSV} disabled={isProcessing} className="w-full py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-bold flex items-center justify-center gap-3 transition-all shadow-md active:scale-[0.99]">
                {isProcessing ? <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Download className="w-6 h-6" />}
                Generar Fitxer Unificat per CUPS
              </button>
            </div>
          </div>
        )}
      </main>
      
      <footer className="mt-auto py-8 opacity-40 text-[10px] text-slate-500 text-center uppercase tracking-widest">
        Processament segur en el costat del client
      </footer>
    </div>
  );
};

export default App;
