export interface ConstantModel {
  translations: Record<string, string>;
}

export interface ConstantState {
  constants: ConstantModel | null;
  loading: boolean;
  error: string | null;
}
