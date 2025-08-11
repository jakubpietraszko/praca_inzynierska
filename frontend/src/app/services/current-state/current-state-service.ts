import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class CurrentStateService {
  private modelSubject = new BehaviorSubject<string | null>(null);
  private datasetSubject = new BehaviorSubject<string | null>(null);

  model$ = this.modelSubject.asObservable();
  dataset$ = this.datasetSubject.asObservable();

  setModel(model: string | null): void {
    this.modelSubject.next(model);
  }

  setDataset(dataset: string | null): void {
    this.datasetSubject.next(dataset);
  }

  getModel(): string | null {
    return this.modelSubject.value;
  }

  getDataset(): string | null {
    return this.datasetSubject.value;
  }

  reset(): void {
    this.modelSubject.next(null);
    this.datasetSubject.next(null);
  }
}
