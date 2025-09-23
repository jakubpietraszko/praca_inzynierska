import { Injectable, signal } from '@angular/core';
import { environment } from '../../environments/environment';
import { Observable, Subject, BehaviorSubject } from 'rxjs';
import { throttleTime } from 'rxjs/operators';

@Injectable({
  providedIn: 'root',
})
export class HeatmapsWebsocket {
  readonly wsUrl = environment.wsUrl;
  private socket: WebSocket | null = null;

  private _dataSubject = new Subject<any>();
  private _isConnectedSubject = new BehaviorSubject<boolean>(false);
  private _errorSubject = new BehaviorSubject<string | null>(null);

  readonly data$ = this._dataSubject.asObservable();
  readonly isConnected$ = this._isConnectedSubject.asObservable();
  readonly error$ = this._errorSubject.asObservable();

  connect(): void {
    if (!this.socket || this.socket.readyState === WebSocket.CLOSED) {
      this.socket = new WebSocket(this.wsUrl);

      this.socket.onopen = () => {
        this._isConnectedSubject.next(true);
        this._errorSubject.next(null);
      };

      this.socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this._dataSubject.next({ ...data, timestamp: Date.now() });
      };

      this.socket.onerror = (event) => {
        this._errorSubject.next('WebSocket error occurred');
      };

      this.socket.onclose = (event) => {
        this._isConnectedSubject.next(false);
      };
    }
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this._isConnectedSubject.next(false);
    }
  }
}
