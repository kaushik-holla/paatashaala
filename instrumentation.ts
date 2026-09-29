/** Keep the Node-only startup graph out of the Edge instrumentation bundle. */
export async function register(): Promise<void> {
  if (process.env.NEXT_RUNTIME === 'nodejs') {
    const { registerNode } = await import('./instrumentation.node');
    await registerNode();
  }
}
