
export interface EnergyRecord {
  CUPS: string;
  Fecha: string;
  Hora: string;
  AE_kWh: string;
  AE_AUTOCONS_kWh: string;
}

export interface FileData {
  id: string;
  name: string;
  records: EnergyRecord[];
  cups: string;
}

export interface MergedRow {
  fecha: string;
  hora: string;
  cupsValues: Record<string, { ae: string; autocons: string }>;
}
