export function shouldClearDraftElementReference(
  response: Response,
  sentSelectionVersion: number,
  currentSelectionVersion: number | undefined,
): boolean {
  return (
    response.ok &&
    response.headers.get('X-Paatashaala-Element-Reference-Accepted') === '1' &&
    currentSelectionVersion === sentSelectionVersion
  );
}
