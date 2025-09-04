import { TestBed } from '@angular/core/testing';

import { Heatmaps } from './heatmaps';

describe('Heatmaps', () => {
  let service: Heatmaps;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(Heatmaps);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
