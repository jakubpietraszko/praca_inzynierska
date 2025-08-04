import { TestBed } from '@angular/core/testing';

import { PathsImagesService } from './paths-images-service';

describe('PathsImagesService', () => {
  let service: PathsImagesService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PathsImagesService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
