export interface ConstantModel {
  translations: { [key: string]: string };
}

export interface ConstantState {
  constants: ConstantModel | null;
  loading: boolean;
  error: string | null;
}
