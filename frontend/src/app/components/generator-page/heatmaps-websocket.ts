import { Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';
import { Subject, BehaviorSubject } from 'rxjs';
import { HeatmapNotification } from '../../generated-interfaces/notification';

@Injectable({
  providedIn: 'root',
})
export class HeatmapsWebsocket {
  private socket: WebSocket | null = null;

  private _dataSubject = new Subject<HeatmapNotification>();
  private _isConnectedSubject = new BehaviorSubject<boolean>(false);
  private _errorSubject = new BehaviorSubject<string | null>(null);

  public readonly data$ = this._dataSubject.asObservable();
  public readonly isConnected$ = this._isConnectedSubject.asObservable();
  public readonly error$ = this._errorSubject.asObservable();

  public connect(): void {
    if (!this.socket || this.socket.readyState === WebSocket.CLOSED) {
      this.socket = new WebSocket(environment.wsUrl);

      this.socket.onopen = () => {
        this._isConnectedSubject.next(true);
        this._errorSubject.next(null);
      };

      this.socket.onmessage = (event) => {
        try {
          const data: HeatmapNotification = JSON.parse(event.data);
          this._dataSubject.next(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this._errorSubject.next('WebSocket error occurred');
      };

      this.socket.onclose = () => {
        this._isConnectedSubject.next(false);
      };
    }
  }

  public disconnect(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this._isConnectedSubject.next(false);
    }
  }
}
