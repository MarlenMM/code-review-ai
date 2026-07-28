/**
 * Mocha glue `@vscode/test-electron` calls inside the real extension host
 * (see `runTest.ts`). Deliberately dependency-free file discovery (no
 * `glob` package) -- one flat directory of `*.test.js` files doesn't need
 * a pattern-matching library.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import Mocha from "mocha";

export async function run(): Promise<void> {
  const mocha = new Mocha({ ui: "tdd", color: true, timeout: 30_000 });
  const testsRoot = __dirname;

  fs.readdirSync(testsRoot)
    .filter((f) => f.endsWith(".test.js"))
    .forEach((f) => mocha.addFile(path.resolve(testsRoot, f)));

  return new Promise((resolve, reject) => {
    mocha.run((failures) => {
      if (failures > 0) {
        reject(new Error(`${failures} integration test(s) failed.`));
      } else {
        resolve();
      }
    });
  });
}
