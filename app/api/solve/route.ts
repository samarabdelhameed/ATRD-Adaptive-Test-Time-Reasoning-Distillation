import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';

const execAsync = promisify(exec);

const PROJECT_ROOT = path.join(process.cwd());

export async function POST(request: NextRequest) {
  try {
    const { question, budget = 4096 } = await request.json();

    if (!question || !question.trim()) {
      return NextResponse.json({ error: 'Question is required' }, { status: 400 });
    }

    // Escape the question for shell safety
    const safeQuestion = question.replace(/'/g, "'\\''");

    const { stdout, stderr } = await execAsync(
      `python3 scripts/solve_question.py '${safeQuestion}' ${budget}`,
      {
        cwd: PROJECT_ROOT,
        timeout: 30000, // 30 second timeout
      }
    );

    if (stderr && !stdout) {
      console.error('Python stderr:', stderr);
      return NextResponse.json(
        { error: 'Backend error', message: stderr },
        { status: 500 }
      );
    }

    const result = JSON.parse(stdout.trim());

    if (result.error) {
      return NextResponse.json({ error: result.error }, { status: 500 });
    }

    return NextResponse.json(result);
  } catch (error: any) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: 'Backend unavailable', message: String(error?.message || error) },
      { status: 500 }
    );
  }
}
