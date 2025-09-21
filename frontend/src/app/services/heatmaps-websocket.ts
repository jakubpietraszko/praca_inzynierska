import { Injectable, signal } from '@angular/core';
import { environment } from '../../environments/environment';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class HeatmapsWebsocket {
  readonly wsUrl = environment.wsUrl;
  private socket: WebSocket | null = null;

  private _data = signal<any>(null);
  private _isConnected = signal(false);
  private _error = signal<string | null>(null);

  readonly data = this._data.asReadonly();
  readonly isConnected = this._isConnected.asReadonly();
  readonly error = this._error.asReadonly();

  connect(): void {
    if (!this.socket || this.socket.readyState === WebSocket.CLOSED) {
      this.socket = new WebSocket(this.wsUrl);

      this.socket.onopen = () => {
        console.log('WebSocket connection established');
        this._isConnected.set(true);
        this._error.set(null);
      };

      this.socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this._data.set(data);
        console.log('Received data:', data);
      };

      this.socket.onerror = (event) => {
        console.error('WebSocket error:', event);
        this._error.set('WebSocket error occurred');
      };

      this.socket.onclose = (event) => {
        console.log('WebSocket connection closed:', event);
        this._isConnected.set(false);
      };
    }
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this._isConnected.set(false);
    }
  }
}
